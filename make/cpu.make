
BUILD_DIR=/home/azureuser/llm-server
CUSTOM_CPU_FLAGS=avx,avx2
ARCH=amd64

COMMON_SRCS := \
	$(wildcard *.c) \
	$(wildcard *.cpp)
COMMON_HDRS := \
	$(wildcard *.h) \
	$(wildcard *.hpp)

$(BUILD_DIR)/llm-server: TARGET_CPU_FLAGS=$(CUSTOM_CPU_FLAGS)
$(BUILD_DIR)/llm-server: ./llama/*.go ./main/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	go env -w "CGO_CFLAGS_ALLOW=-mfma|-mf16c"
	go env -w "CGO_CXXFLAGS_ALLOW=-mfma|-mf16c"
	GOARCH=$(ARCH) go build -buildmode=pie "-ldflags=-w -s -X=version.Version=1.0.0" -trimpath  -tags "$(CUSTOM_CPU_FLAGS)" -o ./llm-server ./main

.PHONY: clean all

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
