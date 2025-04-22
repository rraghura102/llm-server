package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.

import(
	"context"
	"errors"
	"fmt"
	"encoding/json"
	"log/slog"
	"net/http"
)

// embeddings handles the /embeddings endpoint to generate vector embeddings
// for a given input text using the LLM backend.
//
// Workflow:
//   - Decodes the JSON request into an EmbeddingRequest.
//   - Creates a new sequence with embedding-only mode enabled.
//   - Acquires a free sequence slot and loads cache if enabled.
//   - Waits for the embedding to be generated.
//   - Responds with the embedding vector as JSON.
//
// Request example:
// {
//   "content": "What is the capital of France?",
//   "cachePrompt": true
// }
//
// Response example:
// {
//   "embedding": [0.025, -0.132, ...]
// }
//
// This endpoint is useful for tasks like semantic search, similarity matching,
// or downstream ML models requiring text embeddings.
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	slog.Debug("embedding request", "content", req.Content)

	// Initialize an embedding-only sequence
	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{
		embedding: true,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Acquire available sequence slot
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embeddings request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return
	}

	// Assign sequence to the first free slot
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

	// Wait for the embedding to be returned on the channel
	embedding := <-seq.embedding

	// Encode and return the response
	if err := json.NewEncoder(w).Encode(&EmbeddingResponse{
		Embedding: embedding,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}
