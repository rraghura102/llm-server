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

import(
	"context"
	"errors"
	"fmt"
	"log"
	"time"
	"encoding/json"
	"log/slog"
	"net/http"
	"llm-server/llama"
)

// securecompletion handles the /securecompletion endpoint for streaming
// encrypted LLM completions.
//
// It performs the following steps:
//   - Parses the JSON request containing an encrypted prompt and symmetric key.
//   - Decrypts the symmetric key using the server's RSA private key.
//   - Decrypts the actual prompt using the symmetric AES key.
//   - Initializes a new sequence with predefined sampling parameters.
//   - Streams encrypted responses (each encrypted using the same symmetric AES key) to the client.
//
// Notes:
//   - This endpoint enforces secure prompt transmission and encrypted streaming output.
//   - It supports streaming JSON responses via chunked transfer encoding.
//   - It is hardcoded for specific sampling parameters and disables embedding and stop criteria.
//
// Example JSON request:
// {
//   "role": "user",
//   "EncryptedPrompt": "base64-encoded encrypted prompt",
//   "encryptedSymmetricKey": "base64-encoded encrypted AES key"
// }
func (s *Server) securecompletion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Role                 string `json:"role"`
		EncryptedPrompt      string `json:"EncryptedPrompt"`
		EncryptedSymmetricKey string `json:"encryptedSymmetricKey"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	privateKey, exists := KeyStore.Get("privateKey")
	if !exists {
		fmt.Println("Key not found in cache")
	}

	symmetricKey, err := RsaDecrypt(privateKey, req.EncryptedSymmetricKey)
	if err != nil {
		log.Fatal("Error decrypting symmetric key", err)
		return
	}

	prompt, err := AesDecrypt(symmetricKey, req.EncryptedPrompt)
	if err != nil {
		log.Fatal("Error decrypting prompt", err)
		return
	}

	// Set headers for streaming JSON
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Hardcoded sampling parameters for secure completions
	samplingParams := llama.SamplingParams{
		TopK:           40,
		TopP:           0.9,
		MinP:           0,
		TypicalP:       1,
		Temp:           0.8,
		RepeatLastN:    64,
		PenaltyRepeat:  1.1,
		PenaltyFreq:    0,
		PenaltyPresent: 0,
		Mirostat:       0,
		MirostatTau:    5,
		MirostatEta:    0.1,
		PenalizeNl:     true,
		Seed:           uint32(0),
		Grammar:        "false",
	}

	// Create new decoding sequence
	seq, err := s.NewSequence(fmt.Sprintf(promptFormat, prompt), nil, NewSequenceParams{
		numPredict:     -1,
		stop:           nil,
		numKeep:        4,
		samplingParams: &samplingParams,
		embedding:      false,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Acquire available sequence slot
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting securecompletion due to client disconnection")
		} else {
			slog.Error("Failed to acquire sequence slot", "error", err)
		}
		return
	}

	// Load the sequence into the shared sequence pool
	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, true)
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

	// Begin streaming encrypted content
	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case content, ok := <-seq.responses:
			if ok {
				encryptedContent, err := AesEncrypt(symmetricKey, content)
				if err != nil {
					http.Error(w, fmt.Sprintf("Failed to encrypt content: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Content: encryptedContent,
				}); err != nil {
					http.Error(w, fmt.Sprintf("Failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				// Final response with generation metrics
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
					http.Error(w, fmt.Sprintf("Failed to encode final response: %v", err), http.StatusInternalServerError)
				}
				return
			}
		}
	}
}
