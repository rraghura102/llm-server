package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.
//
// This is the main entry point for the LLM inference server.
//
// It performs the following:
//   - Parses command-line flags for server configuration
//   - Loads the LLM model and optional LoRA/vision components
//   - Initializes concurrent sequence queues and caching
//   - Exposes REST endpoints for health checks, completions, embeddings, and encryption utilities
//   - Starts a blocking HTTP server loop bound to the configured port
//
// The server supports token-based completions, encrypted prompt handling, batch embeddings,
// secure transport with AES/RSA, and configurable FlashAttention and multi-GPU execution.

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"regexp"
	"strconv"
	"sync"
	"net/http"
	"golang.org/x/sync/semaphore"
	"llm-server/llama"
)

// main initializes the server, loads the model, sets up routes, and starts the HTTP server.
//
// It also prints the generated RSA public key to the console and stores the private key
// in memory using the global `KeyStore` for encrypted endpoint use.
func main() {

	config := setupFlags()
	server := createServer(config)
	tensorSplitFloats := createTensorSplitFloats(config)
	modelParams := createModelParameters(config, tensorSplitFloats, server)
	
	server.ready.Add(1)
	go server.loadModel(
		modelParams, 
		config.model, 
		config.lpaths, 
		config.ppath, 
		config.kvSize, 
		config.flashAttention, 
		config.threads, 
		config.multiUserCache)

	server.cond = sync.NewCond(&server.mu)
	ctx, _ := context.WithCancel(context.Background())
	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(config.port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.health)
	mux.HandleFunc("/embedding", server.embeddings)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/secure/completion", server.securecompletion)
	mux.HandleFunc("/generate", server.generate)
	mux.HandleFunc("/secure/generate", server.secureGenerate)

	mux.HandleFunc("/aes/key", AesKeyHandler)
	mux.HandleFunc("/aes/encrypt", AesEncryptHandler)
	mux.HandleFunc("/aes/decrypt", AesDecryptHandler)
	mux.HandleFunc("/rsa/keys", RsaKeysHandler)
	mux.HandleFunc("/rsa/encrypt", RsaEncryptHandler)
	mux.HandleFunc("/rsa/decrypt", RsaDecryptHandler)

	httpServer := http.Server{
		Handler: mux,
	}

	privateKey, publicKey, err := RsaKeys()
	if err != nil {
		log.Fatal("Error generating RSA keys", err)
		return
	}

	log.Println("-----BEGIN PUBLIC KEY-----\n" + publicKey + "\n-----END PUBLIC KEY-----")
	KeyStore.Set("privateKey", privateKey)

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}
}

// setupFlags defines and parses all command-line flags for configuring the server,
// including model paths, context sizes, GPU layer settings, and LoRA/vision options.
func setupFlags() *Config {

	//mac-17, cpu-17, h100-160
	gpuLayers := 17
	//mac-12, cpu-3, h100-160
	threads := 12

    config := &Config{}
    flag.StringVar(&config.model, "model", "models/modelfile", "Path to model binary file")
    flag.IntVar(&config.kvSize, "kv-size", 8192, "Context (or KV cache) size")
    flag.IntVar(&config.batchSize, "batch-size", 512, "Batch size")
    flag.IntVar(&config.parallel, "parallel", 4, "Number of sequences to handle simultaneously")
    flag.IntVar(&config.port, "port", 60000, "Port to expose the server on")
    flag.IntVar(&config.mainGPU, "main-gpu", 0, "Main GPU")
    flag.StringVar(&config.tensorSplit, "tensor-split", "", "Fraction of the model to offload to each GPU, comma-separated list of proportions")
    flag.BoolVar(&config.noMmap, "no-mmap", false, "Do not memory-map model (slower load but may reduce pageouts if not using mlock)")
    flag.BoolVar(&config.mlock, "mlock", false, "Force system to keep model in RAM rather than swapping or compressing")
    flag.StringVar(&config.ppath, "mmproj", "", "Path to projector binary file")
    flag.BoolVar(&config.flashAttention, "flash-attn", true, "Enable flash attention")
    flag.BoolVar(&config.multiUserCache, "multiuser-cache", false, "Optimize input cache algorithm for multiple users")
    flag.Var(&config.lpaths, "lora", "Path to lora layer file (can be specified multiple times)")
    flag.IntVar(&config.gpuLayers, "gpu-layers", gpuLayers, "Number of layers to offload to GPU")
    flag.IntVar(&config.threads, "threads", threads, "Number of threads to use during generation")
    flag.Parse()
    return config
}

// createServer creates a new Server instance with the specified batch size, parallelism,
// and internal semaphore pool used to coordinate concurrent sequence execution.
func createServer(config *Config) (*Server) {
	
	return &Server{
		batchSize: config.batchSize,
		parallel:  config.parallel,
		seqs:      make([] *Sequence, config.parallel),
		seqsSem:   semaphore.NewWeighted(int64(config.parallel)),
		status:    ServerStatusLoadingModel,
	}	
}

// createTensorSplitFloats parses the --tensor-split argument and converts it to
// a slice of float32 values used for multi-GPU tensor partitioning.
func createTensorSplitFloats(config *Config) ([]float32) {

	var tensorSplitFloats []float32
	if config.tensorSplit != "" {
		stringFloats := regexp.MustCompile(",").Split(config.tensorSplit, -1)
		tensorSplitFloats = make([]float32, 0, len(stringFloats))
		for _, s := range stringFloats {
			f, _ := strconv.ParseFloat(s, 32)
			tensorSplitFloats = append(tensorSplitFloats, float32(f))
		}
	}

	return tensorSplitFloats
}

// createModelParameters constructs llama.ModelParams using parsed flags and tensor split values.
// It includes a progress callback to update model load status for external monitoring.
func createModelParameters(config *Config, tensorSplitFloats []float32, server *Server) (llama.ModelParams) {

	return llama.ModelParams{
		NumGpuLayers: config.gpuLayers,
		MainGpu:      config.mainGPU,
		UseMmap:      !config.noMmap && config.lpaths.String() == "",
		UseMlock:     config.mlock,
		TensorSplit:  tensorSplitFloats,
		Progress: func(progress float32) {
			server.progress = progress
		},
	}
}