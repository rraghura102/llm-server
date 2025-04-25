# LLM Inference Server

A Proof of Concept (POC) of a lightweight, multi-user capable HTTP server for running completions, embeddings, and secure prompt processing with a [llama.cpp](https://github.com/ggml-org/llama.cpp) based backend. Supports token-based completions, secure completions with AES and RSA encryption, inference generation, and secure inference generation with AES and RSA encryption. The Inference Server demonstrates a Zero Trust implementation where data is encrypted at REST, in TRANSIT, and in PROCESS. This code is optimized for both MAC ARM and Linux running on Nvidia H100 GPU with secure compute architectures. Instructions to compile and deploy on MAC with the mini Llama-3.2-1B LLM model are provided below. Please contact the author for Windows and Nvidia HPC H100 optimized Linux installation instructions.

⚠️ NOTE
This code is not production-ready and is intended solely for proof-of-concept (PoC) and demonstration purposes. It lacks production-grade features such as authentication, request limits, error handling, hardening, and full model lifecycle management.

# Prequisites

1) Go
2) make
3) python3
5) Hugging Face Client

# MAC ARM Build & Run

1) Install Go

```
brew install go
go version
```

2) Install make

```
brew install make
make --version
```

3) Install Python

```
brew install python
python3 --version
pip3 --version
```
4) Install Hugging Face CLI

```
python3 -m pip install -U "huggingface_hub[cli]"
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
huggingface-cli --version
huggingface-cli --help
pip show huggingface_hub
```

5) Clone Repository

```
git clone https://github.com/rraghura102/llm-server.git
```

6) Download Hugging Face Model

```
cd llm-server\models
huggingface-cli download hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF --include "llama-3.2-1b-instruct-q8_0.gguf" --local-dir ./
mv llama-3.2-1b-instruct-q8_0.gguf modelfile
```

7) Build & Run LLM Server

```
cd .. (llm-server root directory)
rm llm-server
rm -rf build
rm go.*

go mod init llm-server
go mod tidy
make -j 18

./llm-server
```

8) Plain Text API's

```
curl --request GET --url http://localhost:60000/health

curl --location 'http://localhost:60000/completion' \
--header 'Content-Type: application/json' \
--data '{"prompt": "star","n_predict": 128}'

curl --location 'http://localhost:60000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "role": "user",
    "prompt": "One line definition of a star"
}'

```

9) Secure API's

```
Copy Public Key from terminal during llm-server start up

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ern7Gq8olPwMXvzuYzDJ4cJbQQt5aTagjYeZ5oayX2x4q6jF8ztDEO4Psk2gA1MvrdMbzCTxBNrnfe++f+VmxGLGvFhqj2WM0EhjNVlaHPherFgT4YbgUd19pExuBW4+KgsAj8p+6mTuFruMldobPJ7W83DYFQtjng/p5xgu1C5F3R+KqMb+na2/c8bSXAUgM/Eu2h2TDdEISou2vjNswMloxaKPbOC90w6Ty11sjLhf08MM1U3GT5lirAE2XwnPLBA2NAAIV+0UwO9zG7l7Ok+KyWeffKQ2MUjKyfaAl362tgRNd3oT/Hj/AGRK7A5bOKtrs874xFmKJfYUi9ZGwIDAQAB
-----END PUBLIC KEY-----

curl --location 'http://localhost:60000/secure/generate' \
--header 'Content-Type: application/json' \
--data '{
    "role": "user",
    "encryptedPrompt": "LXR6rTcOLai1+3S4wW0iihL64l9FZSNex1H8v14X4+0=",
    "encryptedSymmetricKey": "nstZi4/jGjSnnwGO06ycMOpD0GeAv202Hw9fwi7VG7WxoXNVy+U6c0HVbqnaGdRk2OmpEN9OJcPzG23GGeA51HuQJ3HtsgfqbHTMsZZGu2pzchA0MQBh5Cae631xIRQ8Mc0Ubxj3Rx3JAaxWBAOIP/pcFQb86W3jJqgYoBKtqZ/WcJKRAHzYBM62g5fsJdgOKTen+vjoFxT/oxHjK44aAd9ga3/KrDKvyDwqJ0/8PVxgBid3MdY2W+l1i5dTWlRYnrRnxwQDwkd01pkaKp8tYW8+mVAq36k2grDJqxoZCTs+rH9wnwcNdnej+rkGSe2VjYWUXtbDbO9u0moM0yQUrg=="
}'

curl --location 'http://localhost:60000/secure/completion' \
--header 'Content-Type: application/json' \
--data '{
    "role": "user",
    "encryptedPrompt": "uL6ACZSHqhTcVOA9VFNrITysr2/WA0huwv4lno+iE5WNXvn4se5gP+6yBI12RttNKvB+BHc8IDQ8seAH+hTmuw==",
    "encryptedSymmetricKey": "BCzwpH0yeM+THBrSknWcAchNOVecuYO+AanaIwPVATOBnOXnNfqFcSOY6GhYDtXwkQulgEVX3on1PyuPG5R00raSr1mDay1zqduTj8D9qm10adw+E+faemonc501KWm2XsgpyrOTWz7UaM7pu680Fd5bcjDNWRnLPHPSY3ctbF94IQ6YQ+npvvDAOrPib8hdBsQoQCti0k9ahbdmkBx1LxgDN5/Fd+Lzi6XgkTS2eipkmq6872w0h//GRoesLRmZfsJvPhrhjYBzVvZcOnmkCXal6wLEfLzi/a4RHzCgZTeBAasXqbbDxwfrQOMhyaBqp7EqPVMnG4FCA/YNcG2oLQ=="
}'

```

10) Utility Crypto API's

These utility API's are only to test the POC and should need published by the server in any environment.

```
curl --location 'http://localhost:60000/aes/key'

curl --location 'http://localhost:60000/aes/encrypt' \
--header 'Content-Type: application/json' \
--data '{
    "aesKey": "F13cXsA4GgAXShNfFHIdD+xjnbleAd7z62Aagbci1zY=",
    "text": "star"
}'

curl --location 'http://localhost:60000/aes/decrypt' \
--header 'Content-Type: application/json' \
--data '{
    "aesKey": "HLn1+LNhnOIhWVfvg7cMo+uy/TCuPmJbXXAkT9S5KVk=",
    "encryptedText": "iEAOSKF6zrh1we0W4Bn1MXSyyvxPag6F+MBNhD0CgpDNyL35kAm+98p3gH0qHQBF0PGHwdaEU1GWOO600RrdlMTMZKJiK1TwFT1nyyg/r/4="
}'

curl --location 'http://localhost:60000/rsa/keys'

curl --location 'http://localhost:60000/rsa/encrypt' \
--header 'Content-Type: application/json' \
--data '{
    "publicKey": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ern7Gq8olPwMXvzuYzDJ4cJbQQt5aTagjYeZ5oayX2x4q6jF8ztDEO4Psk2gA1MvrdMbzCTxBNrnfe++f+VmxGLGvFhqj2WM0EhjNVlaHPherFgT4YbgUd19pExuBW4+KgsAj8p+6mTuFruMldobPJ7W83DYFQtjng/p5xgu1C5F3R+KqMb+na2/c8bSXAUgM/Eu2h2TDdEISou2vjNswMloxaKPbOC90w6Ty11sjLhf08MM1U3GT5lirAE2XwnPLBA2NAAIV+0UwO9zG7l7Ok+KyWeffKQ2MUjKyfaAl362tgRNd3oT/Hj/AGRK7A5bOKtrs874xFmKJfYUi9ZGwIDAQAB",
    "text": "F13cXsA4GgAXShNfFHIdD+xjnbleAd7z62Aagbci1zY="
}'

curl --location 'http://localhost:60000/aes/decrypt' \
--header 'Content-Type: application/json' \
--data '{
    "aesKey": "HLn1+LNhnOIhWVfvg7cMo+uy/TCuPmJbXXAkT9S5KVk=",
    "encryptedText": "iEAOSKF6zrh1we0W4Bn1MXSyyvxPag6F+MBNhD0CgpDNyL35kAm+98p3gH0qHQBF0PGHwdaEU1GWOO600RrdlMTMZKJiK1TwFT1nyyg/r/4="
}'
```

### License

© 2025 Rayan Raghuram. All rights reserved.
This project is licensed under the MIT License.
