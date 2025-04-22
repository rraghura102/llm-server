package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.

import(
	"context"
	"errors"
	"fmt"
    "log"
	"strings"
	"time"
	"encoding/json"
	"log/slog"
	"net/http"
	"llm-server/llama"
)

// secureGenerate handles the `/secureGenerate` endpoint for generating a 
// secure, full-model response using encrypted inputs.
//
// Workflow:
//   1. Accepts a JSON request with:
//      - `EncryptedPrompt`: base64 AES-encrypted user prompt
//      - `encryptedSymmetricKey`: RSA-encrypted AES key
//   2. Decrypts the symmetric key using the server's private RSA key
//   3. Decrypts the user prompt using the symmetric AES key
//   4. Applies a formatted system/user/assistant prompt structure
//   5. Uses hardcoded decoding parameters to create a new inference sequence
//   6. Acquires a free sequence slot, loads KV cache, and begins streaming
//   7. Collects all content into memory and sends a full JSON response
//
// Request format:
// {
//   "role": "user",
//   "EncryptedPrompt": "<base64-AES-encrypted string>",
//   "encryptedSymmetricKey": "<base64-RSA-encrypted key>"
// }
//
// Response format:
// {
//   "message": {
//     "role": "assistant",
//     "content": "Decrypted and generated content..."
//   },
//   "model": "llama3.2:3b",
//   "created_at": "2025-04-22T14:12:00Z",
//   "done_reason": "stop",
//   "done": true,
//   "total_duration": 123456789,
//   "load_duration": 4567890,
//   "prompt_eval_count": -1,
//   "prompt_eval_duration": 4567890,
//   "eval_count": 42,
//   "eval_duration": 118888899
// }
//
// Notes:
// - All encryption/decryption is handled server-side before model invocation
// - Prompt formatting is fixed using a system instruction template
// - Response timing is measured and included in the output
func (s *Server) secureGenerate(w http.ResponseWriter, r *http.Request) {
    
    var req struct {
    	Role    string `json:"role"` 
        EncryptedPrompt string `json:"EncryptedPrompt"`
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

    prompt,err := AesDecrypt(symmetricKey, req.EncryptedPrompt)
    if err != nil {
        log.Fatal("Error decrypting prompt", err)
        return
    }

    w.Header().Set("Content-Type", "application/json")

    // Hard-code all the parameters as specified
    samplingParams := llama.SamplingParams{
        TopK:             40,
        TopP:             0.9,
        MinP:             0,
        TypicalP:         1,
        Temp:             0.8,
        RepeatLastN:      64,
        PenaltyRepeat:    1.1,
        PenaltyFreq:      0,
        PenaltyPresent:   0,
        Mirostat:         0,
        MirostatTau:      5,
        MirostatEta:      0.1,
        PenalizeNl:       true,
        Seed:             uint32(0),
        Grammar:          "false", 
    }

    seq, err := s.NewSequence(fmt.Sprintf(promptFormat, prompt), nil, NewSequenceParams{
        numPredict:     -1, // Hard-coded as specified
        stop:           nil,
        numKeep:        4,
        samplingParams: &samplingParams,
        embedding:      false,
    })

    if err != nil {
        http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
        return
    }

    // Ensure there is a place to put the sequence, released when removed from s.seqs
    if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
        if errors.Is(err, context.Canceled) {
            slog.Info("Aborting completion request due to client closing the connection")
        } else {
            slog.Error("Failed to acquire semaphore", "error", err)
        }
        return
    }

    s.mu.Lock()
    found := false
    for i, sq := range s.seqs {
        if sq == nil {
            seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, true) // Always using cache_prompt as true
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

    // Use strings.Builder for efficient string concatenation
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
                // Send the final response after collecting all content
                finalContent := strings.TrimSpace(contentBuilder.String())

                // Prepare the response with the required fields
                type Response struct {
                    Message struct {
                        Role    string `json:"role"`
                        Content string `json:"content"`
                    } `json:"message"`
                    Model               string `json:"model"`
                    CreatedAt           string `json:"created_at"`
                    DoneReason          string `json:"done_reason"`
                    Done                bool   `json:"done"`
                    TotalDuration       int64  `json:"total_duration"`
                    LoadDuration        int64  `json:"load_duration"`
                    PromptEvalCount     int    `json:"prompt_eval_count"`
                    PromptEvalDuration  int64  `json:"prompt_eval_duration"`
                    EvalCount           int    `json:"eval_count"`
                    EvalDuration        int64  `json:"eval_duration"`
                }

                response := Response{
                    Model:      "llama3.2:3b",
                    CreatedAt:  time.Now().UTC().Format(time.RFC3339),
                    DoneReason: "stop",
                    Done:       true,
                    TotalDuration: time.Since(seq.startProcessingTime).Nanoseconds(),
                    LoadDuration:  seq.startGenerationTime.Sub(seq.startProcessingTime).Nanoseconds(),
                    PromptEvalCount: -1, // Hard-coded as specified
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