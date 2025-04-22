package main

import (
	"fmt"
	"llm-server/llama"
)

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.
//
// loadModel initializes the LLM model and supporting runtime structures
// including context, image encoder (if available), LoRA layers, and KV cache.
//
// Parameters:
//   - params: model initialization parameters (e.g., context size, F16/BF16 flags)
//   - mpath: path to the base model file (e.g., .gguf or .bin)
//   - lpath: optional list of LoRA adapter paths to be applied after model load
//   - ppath: optional path to a multi-modal vision model (e.g., CLIP, mLLaMA)
//   - kvSize: total size of the KV cache in tokens
//   - flashAttention: whether to enable FlashAttention backend
//   - threads: number of CPU threads to use
//   - multiUserCache: whether to enable multi-user context caching
func (server *Server) loadModel(
	params llama.ModelParams, 
	mpath string, 
	lpath multiLPath, 
	ppath string, 
	kvSize int, 
	flashAttention bool, 
	threads int, 
	multiUserCache bool) {

	initBackend()
	loadModelFromFile(server, mpath, params)
	ctxParams := createContextParameters(server, kvSize, threads, flashAttention)
	setContextWithModel(server, ctxParams)
	applyLoraFromFile(server, lpath, 1.0, threads)
	setImageContext(server, ppath)
	setInputCache(server, kvSize, multiUserCache)
	server.status = ServerStatusReady
	server.ready.Done()
}

// initBackend initializes low-level LLM backend (e.g., llama.cpp internal state).
func initBackend() {
	llama.BackendInit()
}

// loadModelFromFile loads the model from the given path using the provided parameters.
// Errors are printed (but not returned), and stored in `server.model`.
func loadModelFromFile(server *Server, mpath string, params llama.ModelParams) {
	var err error
    server.model, err = llama.LoadModelFromFile(mpath, params)
    if err != nil {
        fmt.Errorf("failed to load model from file: %w", err)
    }
}

// createContextParameters returns a llama.ContextParams object
// based on batch size, KV cache size, parallel sessions, and threading.
func createContextParameters(server *Server, kvSize int, threads int, flashAttention bool) (llama.ContextParams) {
	noOfContexts := kvSize
	batchSize := server.batchSize * server.parallel
	noOfMaxSequences := server.parallel
	return llama.NewContextParams(noOfContexts, batchSize, noOfMaxSequences, threads, flashAttention, "")
}

// setContextWithModel creates a llama.Context instance tied to the loaded model
// using the specified context parameters. Panics if initialization fails.
func setContextWithModel(server *Server, ctxParams llama.ContextParams) {
	var err error
	server.lc, err = llama.NewContextWithModel(server.model, ctxParams)
	if err != nil {
		fmt.Errorf("failed to create new context with model: %w", err)
		panic(err)
	}
}

// applyLoraFromFile loads and applies LoRA adapters (if any) to the current model.
// Each path in `lpath` is applied with a scaling factor and parallel threads.
func applyLoraFromFile(server *Server, lpath multiLPath, scale float32, threads int) {
	if lpath.String() != "" {
		for _, path := range lpath {
			err := server.model.ApplyLoraFromFile(server.lc, path, 1.0, threads)
			if err != nil {
				fmt.Errorf("failed to apply lora from file: %w", err)
				panic(err)
			}
		}
	}
}

// setImageContext loads an image embedding model (e.g., CLIP or mLLaMA) for multi-modal support.
// Panics if the model cannot be initialized from the given path.
func setImageContext(s *Server, ppath string) {
	if ppath != "" {
		var err error
		s.image, err = NewImageContext(s.lc, ppath)
		if err != nil {
			fmt.Errorf("failed to create new image context: %w", err)
			panic(err)
		}
	}
}

// setInputCache creates the input token cache for each user/session
// based on KV size and concurrency configuration.
// Panics if allocation fails.
func setInputCache(s *Server, kvSize int, multiUserCache bool) {
	var err error
	s.cache, err = NewInputCache(s.lc, kvSize, s.parallel, multiUserCache)
	if err != nil {
		fmt.Errorf("failed to create new input cache: %w", err)
		panic(err)
	}
}