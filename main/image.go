package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
	"hash/maphash"
	"log/slog"
	"llm-server/llama"
)

const imageCacheSize = 4

type ImageContext struct {
	mu sync.Mutex
	clip   *llama.ClipContext
	mllama *llama.MllamaContext
	images    []imageCache
	imageHash maphash.Hash
}

type imageCache struct {
	key      uint64
	val      [][]float32
	lastUsed time.Time
}

// NewImageContext initializes an ImageContext for a vision model (clip or mllama).
// It returns an error if the model architecture cannot be determined or is unsupported.
func NewImageContext(llamaContext *llama.Context, modelPath string) (*ImageContext, error) {
	arch, err := llama.GetModelArch(modelPath)
	if err != nil {
		return nil, fmt.Errorf("unable to determine vision architecture: %w (%s)", err, modelPath)
	}

	var c ImageContext
	if arch == "clip" {
		c.clip, err = llama.NewClipContext(llamaContext, modelPath)
	} else if arch == "mllama" {
		c.mllama, err = llama.NewMllamaContext(llamaContext, modelPath)
	} else {
		return nil, fmt.Errorf("unknown vision model architecture: %s", arch)
	}

	if err != nil {
		return nil, err
	}

	c.images = make([]imageCache, imageCacheSize)

	return &c, nil
}

// BatchSize returns the appropriate image batch size depending on the backend model.
// For MLLama (which uses large single-token embeddings), the batch size is always 1.
// For CLIP, it returns the configured batch size.
func (c *ImageContext) BatchSize(configuredBatchSize int) int {
	// If images are not supported, we don't need to allocate embedding batches
	if c == nil {
		return 0
	}

	// Mllama maps an image to 1 embedding token (llava creates many tokens)
	// and doesn't support more than a single image per request.
	// The embeddings are large (100 MB), so allocating a big batch can fail
	// on some systems
	if c.mllama != nil {
		return 1
	}

	return configuredBatchSize
}

// EmbedSize returns the dimensionality of the image embeddings for the active vision model.
func (c *ImageContext) EmbedSize(llamaContext *llama.Context) int {
	if c != nil && c.mllama != nil {
		return c.mllama.EmbedSize(llamaContext)
	} else {
		return llamaContext.Model().NEmbd()
	}
}

// NeedCrossAttention determines whether any input tokens require image-text cross-attention.
// This is only true when MLLama-style image embeddings are present.
func (c *ImageContext) NeedCrossAttention(inputs ...input) bool {
	if c == nil || c.mllama == nil {
		return false
	}

	return slices.ContainsFunc(inputs, func(input input) bool {
		return input.embed != nil
	})
}

// NewEmbed generates image embeddings for the given image data.
// It uses the internal cache to avoid recomputation and delegates to the underlying
// vision model for embedding generation if not cached.
func (c *ImageContext) NewEmbed(llamaContext *llama.Context, data []byte, aspectRatioId int) ([][]float32, error) {
	if c == nil {
		return nil, nil
	}

	if len(data) <= 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data)

	c.mu.Lock()
	defer c.mu.Unlock()

	embed, err := c.findImage(hash)
	if err != nil {
		if c.mllama != nil {
			embed, err = c.mllama.NewEmbed(llamaContext, data, aspectRatioId)
			if err != nil {
				return nil, err
			}
		} else if c.clip != nil {
			embed, err = c.clip.NewEmbed(llamaContext, data)
			if err != nil {
				return nil, err
			}
		} else {
			return nil, errors.New("received image but vision model not loaded")
		}

		c.addImage(hash, embed)
	}

	return embed, nil
}

// hashImage computes a 64-bit hash of the raw image bytes using `maphash` for efficient lookup.
func (c *ImageContext) hashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
}

var errImageNotFound = errors.New("image not found in cache")

func (c *ImageContext) findImage(hash uint64) ([][]float32, error) {
	for i := range c.images {
		if c.images[i].key == hash {
			slog.Debug("loading image embeddings from cache", "entry", i)
			c.images[i].lastUsed = time.Now()
			return c.images[i].val, nil
		}
	}

	return nil, errImageNotFound
}

// findImage attempts to locate an embedding in the cache using the hashed image key.
// If found, it updates the lastUsed timestamp and returns the cached value.
func (c *ImageContext) addImage(hash uint64, embed [][]float32) {
	best := time.Now()
	var bestImage int

	for i := range c.images {
		if c.images[i].key == hash {
			bestImage = i
			break
		}

		if c.images[i].lastUsed.Compare(best) < 0 {
			best = c.images[i].lastUsed
			bestImage = i
		}
	}

	slog.Debug("storing image embeddings in cache", "entry", bestImage, "used", c.images[bestImage].lastUsed)
	c.images[bestImage].key = hash
	c.images[bestImage].val = embed
	c.images[bestImage].lastUsed = time.Now()
}