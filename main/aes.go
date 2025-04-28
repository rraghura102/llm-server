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
	"fmt"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
)

// Response structure for AES key generation
type AesKeyResponse struct {
	AesKey string `json:"aesKey"`
}

// Request structure for encryption
type AesEncryptRequest struct {
	AesKey string `json:"aesKey"`
	Text   string `json:"text"`
}

// Response structure for encryption
type AesEncryptResponse struct {
	EncryptedText string `json:"encryptedText"`
}

// Request structure for decryption
type AesDecryptRequest struct {
	AesKey        string `json:"aesKey"`
	EncryptedText string `json:"encryptedText"`
}

// Response structure for decryption
type AesDecryptResponse struct {
	Text string `json:"text"`
}

// Handles GET requests to generate a new AES key
func AesKeyHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	aesKey, err := AesKey()
	if err != nil {
		http.Error(w, "Error generating AES key", http.StatusInternalServerError)
		return
	}

	response := AesKeyResponse{AesKey: aesKey}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Handles POST requests to encrypt text with the provided AES key
func AesEncryptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request AesEncryptRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	encryptedText, err := AesEncrypt(request.AesKey, request.Text)
	if err != nil {
		http.Error(w, "Error encrypting text", http.StatusInternalServerError)
		return
	}

	response := AesEncryptResponse{EncryptedText: encryptedText}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Handles POST requests to decrypt text with the provided AES key
func AesDecryptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request AesDecryptRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	text, err := AesDecrypt(request.AesKey, request.EncryptedText)
	if err != nil {
		http.Error(w, "Error decrypting text", http.StatusInternalServerError)
		return
	}

	response := AesDecryptResponse{Text: text}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Generates a new random AES-256 key and returns it base64 encoded
func AesKey() (string, error) {
	key := make([]byte, 32) // 32 bytes = 256 bits
	_, err := rand.Read(key)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(key), nil
}

// Encrypts plaintext using AES-256 in CBC mode with PKCS#7 padding
func AesEncrypt(base64Key string, text string) (string, error) {
	aesKey, err := base64.StdEncoding.DecodeString(base64Key)
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return "", err
	}

	paddedText := pad([]byte(text), aes.BlockSize)

	// Generate a new random IV
	iv := make([]byte, aes.BlockSize)
	_, err = rand.Read(iv)
	if err != nil {
		return "", err
	}

	ciphertext := make([]byte, len(paddedText))
	mode := cipher.NewCBCEncrypter(block, iv)
	mode.CryptBlocks(ciphertext, paddedText)

	// Prepend IV to ciphertext
	ciphertextWithIV := append(iv, ciphertext...)
	return base64.StdEncoding.EncodeToString(ciphertextWithIV), nil
}

// Decrypts base64 encoded ciphertext using AES-256 in CBC mode with PKCS#7 unpadding
func AesDecrypt(base64key string, encryptedText string) (string, error) {
	aesKey, err := base64.StdEncoding.DecodeString(base64key)
	if err != nil {
		return "", err
	}

	ciphertextWithIV, err := base64.StdEncoding.DecodeString(encryptedText)
	if err != nil {
		return "", err
	}

	if len(ciphertextWithIV) < aes.BlockSize {
		return "", fmt.Errorf("ciphertext too short")
	}

	// Extract IV and ciphertext
	iv := ciphertextWithIV[:aes.BlockSize]
	ciphertext := ciphertextWithIV[aes.BlockSize:]

	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return "", err
	}

	text := make([]byte, len(ciphertext))
	mode := cipher.NewCBCDecrypter(block, iv)
	mode.CryptBlocks(text, ciphertext)

	text = unpad(text, aes.BlockSize)
	return string(text), nil
}

// Applies PKCS#7 padding
func pad(data []byte, blockSize int) []byte {
	padding := blockSize - len(data)%blockSize
	paddedData := append(data, make([]byte, padding)...)
	for i := len(data); i < len(paddedData); i++ {
		paddedData[i] = byte(padding)
	}
	return paddedData
}

// Removes PKCS#7 padding
func unpad(data []byte, blockSize int) []byte {
	padding := int(data[len(data)-1])
	if padding > blockSize {
		return data
	}
	return data[:len(data)-padding]
}
