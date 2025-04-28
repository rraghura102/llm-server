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
import (
	"errors"
	"fmt"
	"reflect"
	"time"
	"log/slog"
	"llm-server/llama"
)

// InputCache holds a pool of KV cache slots for reusing model context across requests.
type InputCache struct {
	numCtx         int
	slots          []InputCacheSlot
	multiUserCache bool
	lc             *llama.Context
}

// InputCacheSlot represents a single KV cache slot, including cached input,
// usage status, and timestamp of last access.
type InputCacheSlot struct {
	Id       int
	Inputs   []input
	InUse    bool
	lastUsed time.Time
}

// NewInputCache initializes a new input cache with specified size and slot count.
func NewInputCache(lc *llama.Context, kvSize int, numSlots int, multiUserCache bool) (*InputCache, error) {
	if kvSize/numSlots < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)
	for i := range slots {
		slots[i] = InputCacheSlot{
			Id:     i,
			Inputs: make([]input, 0),
		}
	}

	return &InputCache{
		numCtx:         kvSize / numSlots,
		slots:          slots,
		multiUserCache: multiUserCache,
		lc:             lc,
	}, nil
}

// ShiftCacheSlot removes old inputs from a slot if the total cached tokens exceed context size.
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	discard := c.ShiftDiscard(len(slot.Inputs), numKeep)
	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if !c.lc.KvCacheSeqRm(slot.Id, numKeep, numKeep+discard) {
		return fmt.Errorf("unable to remove old kv cache entries (id: %v, keep: %v discard: %v)", slot.Id, numKeep, discard)
	}
	c.lc.KvCacheSeqAdd(slot.Id, numKeep+discard, len(slot.Inputs), -discard)

	for i := numKeep + discard; i < len(slot.Inputs); i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:len(slot.Inputs)-discard]

	return nil
}

// ShiftDiscard computes how many tokens need to be discarded to meet target free space in the context.
func (c *InputCache) ShiftDiscard(inputLen int, numKeep int) int {
	targetFree := (c.numCtx - numKeep) / 2
	targetFree = max(targetFree, 1)

	currentFree := c.numCtx - inputLen
	discard := targetFree - currentFree

	if discard < 0 {
		discard = 0
	}
	return discard
}

// LoadCacheSlot selects the best available cache slot for the given prompt,
// trims reused tokens, and prepares the slot for inference.
func (c *InputCache) LoadCacheSlot(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, error) {
	var slot *InputCacheSlot
	var numPast int
	var err error

	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == len(prompt) {
		numPast-- // ensure we keep one input to allow sampling
	}

	if !c.lc.KvCacheSeqRm(slot.Id, numPast, -1) {
		// fallback for models not supporting partial erasure
		c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		numPast = 0
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", len(prompt)-numPast)

	prompt = prompt[numPast:]
	slot.Inputs = slot.Inputs[:numPast]

	return slot, prompt, nil
}

// findLongestCacheSlot returns the slot with the longest matching prefix to the prompt.
func (c *InputCache) findLongestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

// findBestCacheSlot returns a cache slot that either matches the longest prefix or is least recently used.
func (c *InputCache) findBestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
		if s.lastUsed.Before(oldest) && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == len(longestSlot.Inputs) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])

		if c.lc != nil {
			c.lc.KvCacheSeqRm(oldestSlot.Id, 0, -1)
			c.lc.KvCacheSeqCp(longestSlot.Id, oldestSlot.Id, 0, longest)
		}
	}

	return oldestSlot, longest, nil
}

// countCommonPrefix returns the number of matching elements from the start of two input slices.
func countCommonPrefix(a []input, b []input) int {
	var count int
	for i := range a {
		if i >= len(b) || !reflect.DeepEqual(a[i], b[i]) {
			break
		}
		count++
	}
	return count
}
