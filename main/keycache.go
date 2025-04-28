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
	"sync"
)

// KeyCache provides a thread-safe in-memory key-value store
// used for managing sensitive values such as encryption keys
// (e.g., RSA private keys, AES symmetric keys) in LLM server sessions.
//
// It supports concurrent read and write access via a read-write mutex,
// ensuring low-latency safe reads while allowing exclusive writes.
//
// The global variable `KeyStore` can be used as a singleton instance
// throughout the server for temporary key caching.
type KeyCache struct {
	store map[string]string
	mutex sync.RWMutex
}

// NewKeyCache initializes and returns a new KeyCache instance.
func NewKeyCache() *KeyCache {
	return &KeyCache{
		store: make(map[string]string),
	}
}

var KeyStore = NewKeyCache()

// Set stores a key-value pair in the cache with write-lock protection.
// It overwrites the value if the key already exists.
func (c *KeyCache) Set(key, value string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.store[key] = value
}

// Get retrieves a value for the given key with read-lock protection.
// It returns the value and a boolean indicating if the key was found.
func (c *KeyCache) Get(key string) (string, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	val, exists := c.store[key]
	return val, exists
}