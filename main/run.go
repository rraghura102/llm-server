package main

/**
 *
 * MIT License
 *
 * Copyright (c) 2025 Rayan Raghuram
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// This module contains the main loop (`run`) responsible for managing and executing
// LLM inference sequences. It handles:
//   - Creating token and embedding batches
//   - Coordinating token decoding via llama.cpp
//   - Managing input/output queues for concurrent sequence processing
//   - Handling prompt limits, embedding-only sequences, stop sequences, and Unicode safety
//
// This engine supports multi-session inference, caching, and efficient token streaming.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"log/slog"
	"unicode/utf8"
	"llm-server/llama"
)

// run continuously processes queued sequences, executing batched decoding
// for either tokens or image embeddings, until the context is cancelled.
func (server *Server) run(ctx context.Context) {
	
	server.ready.Wait()

	tokenBatch := createTokenBatch(server)
	defer tokenBatch.Free()
	embedBatch := createEmbedBatch(server)
	defer embedBatch.Free()

	for {
		select {
			case <-ctx.Done():
				return
			default:
				err := processBatch(server, tokenBatch, embedBatch)
				if err != nil {
					panic(err)
				}

				tokenBatch.Clear()
				embedBatch.Clear()
		}
	}
}

// createTokenBatch creates a new llama.Batch instance used for token decoding
// across active sequences. Panics if allocation fails.
func createTokenBatch(server *Server) *llama.Batch {

	tokenBatch, err := llama.NewBatch(server.batchSize, len(server.seqs), 0)
	if err != nil {
		panic(err)
	}

	return tokenBatch
}

// createEmbedBatch creates a batch for image embeddings (used in multi-modal inputs).
// Returns an empty placeholder if image support is disabled or batch size is zero.
func createEmbedBatch(server *Server) *llama.Batch {

	var err error
	var embedBatch *llama.Batch
	embedBatchSize := server.image.BatchSize(server.batchSize)
	if embedBatchSize != 0 {
		embedBatch, err = llama.NewBatch(embedBatchSize, len(server.seqs), server.image.EmbedSize(server.lc))
		if err != nil {
			panic(err)
		}
	} else {
		embedBatch = &llama.Batch{}
	}

	return embedBatch
}

// processBatch gathers sequences into a batch, decodes them using llama.cpp,
// and performs sampling, stop detection, and response flushing.
//
// It alternates between embedding and token batches based on the input type.
// Handles embedding-only outputs, stop-sequence truncation, and streaming output.
//
// Returns an error only in case of decoding failure or KV cache mismanagement.
func processBatch(s *Server, tokenBatch *llama.Batch, embedBatch *llama.Batch) error {

	s.mu.Lock()
	for allNil(s) {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	var batch *llama.Batch
	crossAttention := false

	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]

		if seq == nil {
			continue
		}

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			removeSequence(s, seqIdx, "limit")
			continue
		}

		for i, input := range seq.inputs {
			if len(seq.cache.Inputs)+len(seq.pendingInputs)+1 > s.cache.numCtx {
				if len(seq.pendingInputs) == 0 {
					err := s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
					if err != nil {
						return err
					}
				} else {
					break
				}
			}

			embedding := input.embed != nil

			// If we don't currently have a batch, use one of the correct type and
			// fill it up as much as possible across all sequences. If we encounter an
			// input of the opppsite type, stop for that sequence but then pick up from
			// there for the next batch, ensuring that we alternate types
			if batch == nil {
				if !embedding {
					batch = tokenBatch
				} else {
					batch = embedBatch
					seq.crossAttention = s.image.NeedCrossAttention(input)
				}
			} else if embedding != batch.IsEmbedding() || crossAttention != seq.crossAttention {
				s.nextSeq = seqIdx
				break
			}

			if i >= batch.Size() {
				break
			}

			crossAttention = seq.crossAttention
			batch.Add(input.token, input.embed, len(seq.cache.Inputs)+len(seq.pendingInputs), i+1 == len(seq.inputs), seq.cache.Id)
			seq.pendingInputs = append(seq.pendingInputs, input)
			seq.iBatch = batch.NumTokens() - 1
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	if batch == nil || batch.NumTokens() == 0 {
		return nil
	}

	s.lc.SetCrossAttention(crossAttention)

	err := s.lc.Decode(batch)
	if err != nil {
		if errors.Is(err, llama.ErrKvCacheFull) {
			slog.Debug("defragmenting kv cache")
			s.cache.lc.KvCacheDefrag()
			err = s.lc.Decode(batch)
		}
		if err != nil {
			return fmt.Errorf("failed to decode batch: %w", err)
		}
	}

	if crossAttention {
		// synchronize state to ensure the cross attention batch is complete.
		// needed specifically for multi-GPU systems otherwise an inflight
		// task may be incorrectly invalidated causing a crash
		s.lc.Synchronize()
	}

	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		// After calling Decode, pending inputs are now in the cache
		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []input{}
		}

		// don't sample prompt processing
		if len(seq.inputs) != 0 {
			continue
		}

		seq.numDecoded += 1
		if seq.numDecoded == 1 {
			seq.startGenerationTime = time.Now()
		}

		// if done processing the prompt, generate an embedding and return
		if seq.embeddingOnly {
			embed := s.lc.GetEmbeddingsSeq(seq.cache.Id)
			if embed == nil {
				embed = s.lc.GetEmbeddingsIth(seq.iBatch)
			}

			seq.embedding <- embed
			removeSequence(s, i, "")
			continue
		}

		// sample a token
		token := seq.samplingCtx.Sample(s.lc, seq.iBatch)
		seq.samplingCtx.Accept(token, true)
		piece := s.model.TokenToPiece(token)

		seq.numPredicted++

		// if it's an end of sequence token, break
		if s.model.TokenIsEog(token) {
			// TODO (jmorganca): we should send this back
			// as it's important for the /api/generate context
			// seq.responses <- piece

			removeSequence(s, i, "stop")
			continue
		}

		seq.inputs = []input{{token: token}}

		seq.pendingResponses = append(seq.pendingResponses, piece)
		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := findStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = truncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Update the cache based on the tokens that will be returned:
			// - We have 1 token more than is currently in the cache because
			// the last one generated wasn't submitted to Decode
			// - Remove any stop sequences that we stripped out
			// - If truncateStop removed a portion of a token, drop that
			// - As defense-in-depth, if truncatedToken didn't find a stop token
			// remove the extra one that we added to the cache len
			tokenLen := len(seq.cache.Inputs) + 1
			tokenLen -= origLen - newLen
			if tokenTruncated || origLen == newLen {
				tokenLen--
			}
			seq.cache.Inputs = seq.cache.Inputs[:tokenLen]

			removeSequence(s, i, "stop")
			continue
		}

		if containsStopSuffix(sequence, seq.stop) {
			continue
		}

		if incompleteUnicode(sequence) {
			continue
		}

		if !flushPending(seq) {
			removeSequence(s, i, "connection")
		}
	}

	return nil
}

// allNil returns true if no active sequences are in the server's sequence pool.
func allNil(s *Server) bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

// removeSequence finalizes a sequence by marking its reason for completion,
// flushing pending tokens, closing channels, and releasing the cache slot.
func removeSequence(s *Server, seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason
	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
	s.seqsSem.Release(1)
}

// incompleteUnicode checks if the last bytes in a string form an incomplete
// UTF-8 character, helping to avoid sending invalid output mid-sequence.
func incompleteUnicode(token string) bool {
	incomplete := false

	// check if there is incomplete UTF-8 character at the end
	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			// continuation byte: 10xxxxxx
			continue
		}

		if (c & 0xe0) == 0xc0 {
			// 2-byte character: 110xxxxx ...
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			// 3-byte character: 1110xxxx ...
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			// 4-byte character: 11110xxx ...
			incomplete = i < 4
		}

		// else 1-byte character or invalid byte
		break
	}

	return incomplete
}

// flushPending sends all buffered string tokens (`pendingResponses`) as a
// single output string, trimming invalid UTF-8 if present. It returns false
// if the client has disconnected.
func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	seq.pendingResponses = []string{}

	// Check if there are any partial UTF-8 characters remaining.
	// We already check and queue as we are generating but some may
	// still make it here:
	// - Sequence is ending, e.g. generation limit has been hit
	// - Invalid characters in the middle of a string
	// This is a stricter check to ensure we never output invalid Unicode.
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- joined:
		return true
	case <-seq.quit:
		return false
	}
}

// findStop returns true if any configured stop sequence is present in
// the generated output string.
func findStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

// truncateStop trims the output stream up to the first occurrence of a
// stop sequence and rebuilds the token stream back into valid string chunks.
func truncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

	// Split truncated string back into pieces of original lengths
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string
	tokenTruncated := false
	start := 0
	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
			tokenTruncated = true
		}
		result = append(result, joined[start:end])
		start = end
	}

	return result, tokenTruncated
}

// containsStopSuffix returns true if any stop sequence is partially matched
// at the end of the current response (used to delay early flushing).
func containsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}