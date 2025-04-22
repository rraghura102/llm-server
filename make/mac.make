
BUILD_DIR=./
CUSTOM_CPU_FLAGS=
ARCH=arm64

COMMON_SRCS := \
	$(wildcard *.c) \
	$(wildcard *.cpp)
COMMON_HDRS := \
	$(wildcard *.h) \
	$(wildcard *.hpp)

$(BUILD_DIR)/llm-server: TARGET_CPU_FLAGS=$(CUSTOM_CPU_FLAGS)
$(BUILD_DIR)/llm-server: ./llama/*.go ./main/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=$(ARCH) go build -buildmode=pie "-ldflags=-w -s -X=version.Version=1.0.0" -trimpath -o ./llm-server ./main

.PHONY: clean all

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
