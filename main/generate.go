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
	"strings"
	"time"
	"encoding/json"
	"log/slog"
	"net/http"
	"llm-server/llama"
)

// prompt format to passed to the llm
const promptFormat = "<|start_header_id|>system<|end_header_id|>\n\n" + 
    "Cutting Knowledge Date: December 2023\n\n" + 
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + 
    "%s" + 
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

// generate handles the `/generate` endpoint to produce a full LLM response
// for a given user prompt using hardcoded sampling parameters.
//
// Workflow:
//   - Accepts a JSON request with a role and prompt string.
//   - Applies a standardized prompt format that includes role tags and system instructions.
//   - Creates a new sequence with the prompt and predefined decoding parameters.
//   - Acquires a slot for inference and streams the full response into memory.
//   - Sends a structured JSON response that includes metadata and timing information.
//
// Request format:
// {
//   "role": "user",
//   "prompt": "Tell me about quantum physics"
// }
//
// Response format:
// {
//   "message": {
//     "role": "assistant",
//     "content": "Quantum physics is..."
//   },
//   "model": "llama3.2:3b",
//   "created_at": "2025-04-22T13:45:00Z",
//   "done_reason": "stop",
//   "done": true,
//   "total_duration": 123456789,
//   "load_duration": 4567890,
//   "prompt_eval_count": -1,
//   "prompt_eval_duration": 4567890,
//   "eval_count": 52,
//   "eval_duration": 118888899
// }
func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Role   string `json:"role"`
        Prompt string `json:"prompt"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Bad request", http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")

    // Predefined sampling parameters for generation
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
        Seed:           0,
        Grammar:        "false",
    }

    // Format the prompt using system/user/assistant markers
    seq, err := s.NewSequence(fmt.Sprintf(promptFormat, req.Prompt), nil, NewSequenceParams{
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

    // Acquire inference slot
    if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
        if errors.Is(err, context.Canceled) {
            slog.Info("Aborting completion request due to client closing the connection")
        } else {
            slog.Error("Failed to acquire semaphore", "error", err)
        }
        return
    }

    // Assign sequence into the pool
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
        http.Error(w, "Could not find an available sequence", http.StatusInternalServerError)
        return
    }

    // Collect all output content before responding
    var contentBuilder strings.Builder

    for {
        select {
        case <-r.Context().Done():
            close(seq.quit)
            return
        case content, ok := <-seq.responses:
            if ok {
                contentBuilder.WriteString(content)
            } else {
                finalContent := strings.TrimSpace(contentBuilder.String())

                type Response struct {
                    Message struct {
                        Role    string `json:"role"`
                        Content string `json:"content"`
                    } `json:"message"`
                    Model              string `json:"model"`
                    CreatedAt          string `json:"created_at"`
                    DoneReason         string `json:"done_reason"`
                    Done               bool   `json:"done"`
                    TotalDuration      int64  `json:"total_duration"`
                    LoadDuration       int64  `json:"load_duration"`
                    PromptEvalCount    int    `json:"prompt_eval_count"`
                    PromptEvalDuration int64  `json:"prompt_eval_duration"`
                    EvalCount          int    `json:"eval_count"`
                    EvalDuration       int64  `json:"eval_duration"`
                }

                response := Response{
                    Model:              "llama3.2:3b",
                    CreatedAt:          time.Now().UTC().Format(time.RFC3339),
                    DoneReason:         "stop",
                    Done:               true,
                    TotalDuration:      time.Since(seq.startProcessingTime).Nanoseconds(),
                    LoadDuration:       seq.startGenerationTime.Sub(seq.startProcessingTime).Nanoseconds(),
                    PromptEvalCount:    -1,
                    PromptEvalDuration: seq.startGenerationTime.Sub(seq.startProcessingTime).Nanoseconds(),
                    EvalCount:          seq.numDecoded,
                    EvalDuration:       time.Since(seq.startGenerationTime).Nanoseconds(),
                }
                response.Message.Role = "assistant"
                response.Message.Content = finalContent

                if err := json.NewEncoder(w).Encode(response); err != nil {
                    http.Error(w, fmt.Sprintf("Failed to encode final response: %v", err), http.StatusInternalServerError)
                }
                return
            }
        }
    }
}
