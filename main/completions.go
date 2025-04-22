package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"time"
	"encoding/json"
	"log/slog"
	"net/http"
	"llm-server/llama"
)

// completion handles the /completion HTTP endpoint for LLM inference.
//
// It decodes the JSON request body into a CompletionRequest, initializes a
// sampling context and sequence, acquires a slot in the global sequence pool,
// streams completion responses as JSON chunks to the client, and sends timing
// information in the final response.
func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req CompletionRequest
	req.Options = Options(DefaultOptions())
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Map HTTP request params to llama sampling params
	var samplingParams llama.SamplingParams
	samplingParams.TopK = req.TopK
	samplingParams.TopP = req.TopP
	samplingParams.MinP = req.MinP
	samplingParams.TypicalP = req.TypicalP
	samplingParams.Temp = req.Temperature
	samplingParams.RepeatLastN = req.RepeatLastN
	samplingParams.PenaltyRepeat = req.RepeatPenalty
	samplingParams.PenaltyFreq = req.FrequencyPenalty
	samplingParams.PenaltyPresent = req.PresencePenalty
	samplingParams.Mirostat = req.Mirostat
	samplingParams.MirostatTau = req.MirostatTau
	samplingParams.MirostatEta = req.MirostatEta
	samplingParams.PenalizeNl = req.PenalizeNewline
	samplingParams.Seed = uint32(req.Seed)
	samplingParams.Grammar = req.Grammar

	// Create a new decoding sequence
	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:     req.NumPredict,
		stop:           req.Stop,
		numKeep:        req.NumKeep,
		samplingParams: &samplingParams,
		embedding:      false,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Acquire sequence slot
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return
	}

	// Assign sequence to a slot
	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, req.CachePrompt)
			if err != nil {
				s.mu.Unlock()
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}

			seq.crossAttention = s.image.NeedCrossAttention(seq.cache.Inputs...)
			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	// Begin streaming tokens to the client
	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case content, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Content: content,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}
				flusher.Flush()
			} else {
				// Final response with token timings
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Stop:         true,
					StoppedLimit: seq.doneReason == "limit",
					Timings: Timings{
						PromptN:     seq.numPromptInputs,
						PromptMS:    float64(seq.startGenerationTime.Sub(seq.startProcessingTime).Milliseconds()),
						PredictedN:  seq.numDecoded,
						PredictedMS: float64(time.Since(seq.startGenerationTime).Milliseconds()),
					},
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}
				return
			}
		}
	}
}

// NewSequence creates a new sequence object from a prompt and optional images,
// applying context window trimming, caching policies, and sampling configurations.
func (s *Server) NewSequence(prompt string, images []ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	startTime := time.Now()

	inputs, err := inputs(s, prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = len(inputs)
	}
	if s.model.AddBOSToken() {
		params.numKeep += 1
	}
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	// Trim inputs to fit context window
	if len(inputs) > s.cache.numCtx {
		discard := len(inputs) - s.cache.numCtx
		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[params.numKeep+discard:]...)
		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	var sc *llama.SamplingContext
	if params.samplingParams != nil {
		sc, err = llama.NewSamplingContext(s.model, *params.samplingParams)
		if err != nil {
			return nil, err
		}
		for _, input := range inputs {
			if input.embed == nil {
				sc.Accept(input.token, false)
			}
		}
	}

	return &Sequence{
		inputs:              inputs,
		numPromptInputs:     len(inputs),
		startProcessingTime: startTime,
		numPredict:          params.numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
		embedding:           make(chan []float32, 1),
		samplingCtx:         sc,
		embeddingOnly:       params.embedding,
		stop:                params.stop,
		numKeep:             params.numKeep,
	}, nil
}

// inputs tokenizes the prompt and injects image embeddings (if present)
// by parsing [img-n] placeholders and matching them with provided image data.
func inputs(s *Server, prompt string, images []ImageData) ([]input, error) {
	var inputs []input
	var parts []string
	var matches [][]string

	if s.image != nil {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
	} else {
		parts = []string{prompt}
	}

	for i, part := range parts {
		// Tokenize text
		tokens, err := s.lc.Model().Tokenize(part, i == 0, true)
		if err != nil {
			return nil, err
		}
		for _, t := range tokens {
			inputs = append(inputs, input{token: t})
		}

		// Inject image embedding
		if i < len(matches) {
			n, _ := strconv.Atoi(matches[i][1])

			imageIndex := -1
			for j := range images {
				if images[j].ID == n {
					imageIndex = j
					break
				}
			}
			if imageIndex < 0 {
				return nil, fmt.Errorf("invalid image index: %d", n)
			}

			embed, err := s.image.NewEmbed(s.lc, images[imageIndex].Data, images[imageIndex].AspectRatioID)
			if err != nil {
				return nil, err
			}

			for _, e := range embed {
				inputs = append(inputs, input{embed: e})
			}
		}
	}

	return inputs, nil
}
