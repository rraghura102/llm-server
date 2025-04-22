package main

// Author: Rayan Raghuram
// Cpyright @ 2025 Rayan Raghuram. All rights reserved.
//
// This module provides REST API handlers and cryptographic utilities
// for RSA key generation, encryption, and decryption using standard
// Go crypto packages.
//
// Keys are base64-encoded for safe JSON transport, and support is
// included for 2048-bit RSA keys using PKCS1 and PKIX formats.
//
// Use cases include:
//   - Secure symmetric key exchange (e.g., hybrid AES-RSA encryption)
//   - Encrypted LLM prompts or completions
//   - Client-server public key encryption flows

import(
	"fmt"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
)

// RsaKeyResponse represents a response payload containing
// base64-encoded RSA private and public keys.
type RsaKeyResponse struct {
    PrivateKey string `json:"privateKey"`
    PublicKey string `json:"publicKey"`
}

// RsaEncryptRequest is the request payload for encrypting plaintext
// using a provided base64-encoded RSA public key.
type RsaEncryptRequest struct {
    PublicKey string `json:"publicKey"`
    Text string `json:"text"`
}

// RsaEncryptResponse contains the encrypted text encoded in base64.
type RsaEncryptResponse struct {
    EncryptedText string `json:"encryptedText"`
}

// RsaDecryptRequest is the request payload for decrypting RSA-encrypted
// content using a base64-encoded RSA private key.
type RsaDecryptRequest struct {
    PrivateKey string `json:"privateKey"`
    EncryptedText string `json:"encryptedText"`
}

type RsaDecryptResponse struct {
    Text string `json:"text"`
}

// RsaKeysHandler handles GET /rsa/keys
// It generates a new RSA key pair and returns them in base64-encoded format.
//
// Response:
// {
//   "privateKey": "<base64-RSA-private-key>",
//   "publicKey": "<base64-RSA-public-key>"
// }
func RsaKeysHandler(w http.ResponseWriter, r *http.Request) {

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	privateKey, publicKey, err := RsaKeys()
	if err != nil {
		http.Error(w, "Error generating RSA keys", http.StatusInternalServerError)
		return
	}

	response := RsaKeyResponse {
		PrivateKey: privateKey,
		PublicKey: publicKey,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RsaEncryptHandler handles POST /rsa/encrypt
// It encrypts the given plaintext using the provided base64-encoded public key.
//
// Request:
// {
//   "publicKey": "<base64-RSA-public-key>",
//   "text": "hello"
// }
//
// Response:
// {
//   "encryptedText": "<base64-encrypted-bytes>"
// }
func RsaEncryptHandler(w http.ResponseWriter, r *http.Request) {

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request RsaEncryptRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	encryptedText, err := RsaEncrypt(request.PublicKey, request.Text)
	if err != nil {
		http.Error(w, "Error encrypting text", http.StatusInternalServerError)
		return
	}

	response := RsaEncryptResponse {
		EncryptedText: encryptedText,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RsaDecryptHandler handles POST /rsa/decrypt
// It decrypts base64-encoded RSA ciphertext using the provided private key.
//
// Request:
// {
//   "privateKey": "<base64-RSA-private-key>",
//   "encryptedText": "<base64-cipher>"
// }
//
// Response:
// {
//   "text": "hello"
// }
func RsaDecryptHandler(w http.ResponseWriter, r *http.Request) {

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request RsaDecryptRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	text, err := RsaDecrypt(request.PrivateKey, request.EncryptedText)
	if err != nil {
		http.Error(w, "Error decrypting text", http.StatusInternalServerError)
		return
	}

	response := RsaDecryptResponse {
		Text: text,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RsaKeys generates a new 2048-bit RSA private-public key pair,
// returning both as base64-encoded strings.
func RsaKeys() (string, string, error) {

	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return "", "", err
	}

	privateKeyBytes := x509.MarshalPKCS1PrivateKey(privateKey)
	base64PrivateKey := base64.StdEncoding.EncodeToString(privateKeyBytes)

	publicKeyBytes, err := x509.MarshalPKIXPublicKey(&privateKey.PublicKey)
	if err != nil {
		return "", "", err
	}

	base64PublicKey := base64.StdEncoding.EncodeToString(publicKeyBytes)
	return base64PrivateKey, base64PublicKey, nil
}

// RsaEncrypt encrypts plaintext using a base64-encoded RSA public key,
// returning the ciphertext as a base64-encoded string.
func RsaEncrypt(base64PublicKey string, text string) (string, error) {

	rsaPublicKeyBytes, err := base64.StdEncoding.DecodeString(base64PublicKey)
	if err != nil {
		return "", err
	}

	publicKey, err := x509.ParsePKIXPublicKey(rsaPublicKeyBytes)
	if err != nil {
		return "", err
	}

	rsaPublicKey, ok := publicKey.(*rsa.PublicKey)
	if !ok {
		return "", fmt.Errorf("Not a RSA public key")
	}

	encryptedText, err := rsa.EncryptPKCS1v15(rand.Reader, rsaPublicKey, []byte(text))
	if err != nil {
		return "", err
	}

	base64EncryptedText := base64.StdEncoding.EncodeToString(encryptedText)
	return base64EncryptedText, nil
}

// RsaDecrypt decrypts a base64-encoded RSA ciphertext using a
// base64-encoded RSA private key and returns the plaintext.
func RsaDecrypt(base64PrivateKey string, encryptedText string) (string, error) {

	privateKeyBytes, err := base64.StdEncoding.DecodeString(base64PrivateKey)
	if err != nil {
		return "", err
	}

	rsaPrivateKey, err := x509.ParsePKCS1PrivateKey(privateKeyBytes)
	if err != nil {
		return "", fmt.Errorf("error parsing private key: %v", err)
	}

	encryptedBytes, err := base64.StdEncoding.DecodeString(encryptedText)
	if err != nil {
		return "", err
	}

	textBytes, err := rsa.DecryptPKCS1v15(rand.Reader, rsaPrivateKey, encryptedBytes)
	if err != nil {
		return "", err
	}

	return string(textBytes), nil
}