package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.
//
// This file defines the configuration structures, request/response formats,
// and internal server state types used throughout the LLM inference server.
//
// These types are used in HTTP handlers for completions, embeddings, encryption,
// and model runtime control, including batching, KV cache coordination, and stop detection.

import(
	"strings"
	"sync"
	"time"
	"golang.org/x/sync/semaphore"
	"llm-server/llama"
)

// Config holds CLI configuration options used to initialize the server and model.
// These values are populated from flags defined in `main.go`.
type Config struct {
    model          string
    kvSize         int
    batchSize      int
    gpuLayers      int
    threads        int
    parallel       int
    port           int
    mainGPU        int
    tensorSplit    string
    noMmap         bool
    mlock          bool
    ppath          string
    flashAttention bool
    multiUserCache bool
    lpaths         multiLPath
}

// Server represents the global state of the inference engine, including:
// the LLM model, input queue, batching logic, context, and internal cache.
type Server struct {
	ready sync.WaitGroup
	model *llama.Model
	image *ImageContext
	status ServerStatus
	progress float32
	parallel int
	batchSize int
	mu sync.Mutex
	cond *sync.Cond
	lc *llama.Context
	seqs []*Sequence
	seqsSem *semaphore.Weighted
	cache *InputCache
	nextSeq int
}

// Sequence represents one request sequence being handled by the model.
// It tracks state such as predicted tokens, pending inputs, sampled responses, etc.
type Sequence struct {
	iBatch int
	numPredicted int
	inputs []input
	pendingInputs []input
	pendingResponses []string
	cache *InputCacheSlot
	crossAttention bool
	responses chan string
	quit chan bool
	numPredict int
	samplingCtx *llama.SamplingContext
	embedding chan []float32
	stop []string
	numKeep int
	embeddingOnly bool
	doneReason string
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptInputs     int
}

// input is a single unit of model input: either a token (int) or embedding vector.
// Used to build batches for token decoding or vision-based prompts.
type input struct {
	token int
	embed []float32
}

// EmbeddingRequest is used for POST /embedding, sending a prompt and cache flag.
type EmbeddingRequest struct {
	Content     string `json:"content"`
	CachePrompt bool   `json:"cache_prompt"`
}

// EmbeddingResponse contains the vector embedding returned for a given prompt.
type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// NewSequenceParams configures a new sequence with decoding rules,
// such as stop conditions, sampling params, and embedding-only behavior.
type NewSequenceParams struct {
	numPredict     int
	stop           []string
	numKeep        int
	samplingParams *llama.SamplingParams
	embedding      bool
}

// CompletionRequest is used for POST /completion and /secure/completion endpoints.
// It contains the user prompt, optional images, grammar constraints, and decoding options.
type CompletionRequest struct {
	Prompt      string      `json:"prompt"`
	Images      []ImageData `json:"image_data"`
	Grammar     string      `json:"grammar"`
	CachePrompt bool        `json:"cache_prompt"`

	Options
}

// Options defines all model inference settings such as sampling behavior,
// temperature, penalties, and Mirostat tuning values.
type Options struct {
	Runner

	NumKeep          int      `json:"n_keep"`
	Seed             int      `json:"seed"`
	NumPredict       int      `json:"n_predict"`
	TopK             int      `json:"top_k"`
	TopP             float32  `json:"top_p"`
	MinP             float32  `json:"min_p"`
	TFSZ             float32  `json:"tfs_z"`
	TypicalP         float32  `json:"typical_p"`
	RepeatLastN      int      `json:"repeat_last_n"`
	Temperature      float32  `json:"temperature"`
	RepeatPenalty    float32  `json:"repeat_penalty"`
	PresencePenalty  float32  `json:"presence_penalty"`
	FrequencyPenalty float32  `json:"frequency_penalty"`
	Mirostat         int      `json:"mirostat"`
	MirostatTau      float32  `json:"mirostat_tau"`
	MirostatEta      float32  `json:"mirostat_eta"`
	PenalizeNewline  bool     `json:"penalize_nl"`
	Stop             []string `json:"stop"`
}

// Runner defines lower-level execution parameters related to batch size,
// GPU usage, and model memory configuration (e.g., mlock, mmap, low_vram).
type Runner struct {
	NumCtx    int   `json:"num_ctx,omitempty"`
	NumBatch  int   `json:"num_batch,omitempty"`
	NumGPU    int   `json:"num_gpu,omitempty"`
	MainGPU   int   `json:"main_gpu,omitempty"`
	LowVRAM   bool  `json:"low_vram,omitempty"`
	F16KV     bool  `json:"f16_kv,omitempty"` // Deprecated: This option is ignored
	LogitsAll bool  `json:"logits_all,omitempty"`
	VocabOnly bool  `json:"vocab_only,omitempty"`
	UseMMap   *bool `json:"use_mmap,omitempty"`
	UseMLock  bool  `json:"use_mlock,omitempty"`
	NumThread int   `json:"num_thread,omitempty"`
}

// ImageData holds an encoded image (e.g. JPEG or PNG), its ID, and the aspect ratio class.
// Used for multi-modal completion requests.
type ImageData struct {
	Data          []byte `json:"data"`
	ID            int    `json:"id"`
	AspectRatioID int    `json:"aspect_ratio_id"`
}

// DefaultOptions returns a baseline set of decoding options with commonly tuned values.
// These are overridden per request via the `Options` field in CompletionRequest.
func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict: -1,

		// set a minimal num_keep to avoid issues on context shifts
		NumKeep:          4,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TFSZ:             1.0,
		TypicalP:         1.0,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Mirostat:         0,
		MirostatTau:      5.0,
		MirostatEta:      0.1,
		PenalizeNewline:  true,
		Seed:             -1,

		Runner: Runner{
			// options set when the model is loaded
			NumCtx:    2048,
			NumBatch:  512,
			NumGPU:    -1, // -1 here indicates that NumGPU should be set dynamically
			NumThread: 0,  // let the runtime decide
			LowVRAM:   false,
			UseMLock:  false,
			UseMMap:   nil,
		},
	}
}

// CompletionResponse is the streaming or final response returned by the model.
// It includes the generated text, stop flags, timing, and optionally model metadata.
type CompletionResponse struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`

	Model        string  `json:"model,omitempty"`
	Prompt       string  `json:"prompt,omitempty"`
	StoppedLimit bool    `json:"stopped_limit,omitempty"`
	PredictedN   int     `json:"predicted_n,omitempty"`
	PredictedMS  float64 `json:"predicted_ms,omitempty"`
	PromptN      int     `json:"prompt_n,omitempty"`
	PromptMS     float64 `json:"prompt_ms,omitempty"`

	Timings Timings `json:"timings"`
}

// Timings captures performance measurements for prompt and token generation.
type Timings struct {
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
}

// HealthResponse is returned by the /health endpoint to report server readiness and progress.
type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

// multiLPath allows specifying multiple --lora arguments via CLI flags.
type multiLPath []string

func (m *multiLPath) Set(value string) error {
	*m = append(*m, value)
	return nil
}

func (m *multiLPath) String() string {
	return strings.Join(*m, ", ")
}

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusLoadingModel
	ServerStatusError
)

// ServerStatus is an enum representing the current state of the model/server lifecycle.
type ServerStatus int

// ToString converts a ServerStatus value into a user-friendly string for API output.
func (s ServerStatus) ToString() string {
	switch s {
	case ServerStatusReady:
		return "ok"
	case ServerStatusLoadingModel:
		return "loading model"
	default:
		return "server error"
	}
}
