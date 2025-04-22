
export CGO_CFLAGS_ALLOW = -mfma|-mf16c
export CGO_CXXFLAGS_ALLOW = -mfma|-mf16c
export HIP_PLATFORM = amd
export CGO_ENABLED=1

BUILD_DIR := ./llama/build/linux-amd64
BUILD_RUNNERS := ./llama/build/linux-amd64/runners/cuda_v12_avx/llm-server
RUNNERS_BUILD_DIR := ./llama/build/linux-amd64/runners
GPU_LIB_DIR := /usr/local/cuda-12/lib64
DIST_GPU_RUNNER_DEPS_DIR := ./dist/linux-amd64/lib/ollama

GPU_RUNNER_DRIVER_LIB_LINK := -lcuda
GPU_RUNNER_LIBS_SHORT := cublas cudart cublasLt
GPU_COMPILER_CXXFLAGS :=  -Xcompiler -fPIC -D_GNU_SOURCE
GPU_RUNNER_NAME := cuda_v12

GPU_COMPILER := /usr/local/cuda-12/bin/nvcc
GPU_COMPILER_CFLAGS := -Xcompiler -fPIC -D_GNU_SOURCE
GPU_RUNNER_ARCH_FLAGS := --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -DGGML_CUDA_USE_GRAPHS=1
GPU_RUNNER_EXTRA_VARIANT := _avx
GPU_COMPILER_CUFLAGS := -fPIC -Wno-unused-function -std=c++17 -Xcompiler -mavx -t2 \
	-DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
	-DGGML_USE_CUDA=1 -DGGML_SHARED=1 -DGGML_BACKEND_SHARED=1 -DGGML_BUILD=1 \
	-DGGML_BACKEND_BUILD=1 -DGGML_USE_LLAMAFILE -DK_QUANTS_PER_ITERATION=2 \
	-DNDEBUG -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Wno-deprecated-gpu-targets \
	--forward-unknown-to-host-compiler -use_fast_math -I./llama/ -O3

CGO_EXTRA_LDFLAGS := -L"/usr/local/cuda-12/lib64" -L"/usr/local/cuda-12/lib64/stubs" 
SHARED_PREFIX := lib
SHARED_EXT := so
TARGET_CGO_LDFLAGS := 
EXE_EXT := 
lib := 
OBJ_EXT := o
CCACHE := 

COMMON_HDRS := \
	$(wildcard ./llama/*.h) \
	$(wildcard ./llama/*.hpp)

COMMON_SRCS := \
	$(wildcard ./llama/*.c) \
	$(wildcard ./llama/*.cpp)

GPU_RUNNER_HDRS := \
	$(wildcard llama/ggml-cuda/*.cuh)

GPU_RUNNER_SRCS := \
	$(filter-out $(wildcard llama/ggml-cuda/fattn*.cu),$(wildcard llama/ggml-cuda/*.cu)) \
	$(wildcard llama/ggml-cuda/template-instances/mmq*.cu) \
	llama/ggml.c llama/ggml-backend.cpp llama/ggml-alloc.c llama/ggml-quants.c llama/sgemm.cpp llama/ggml-threading.cpp

GPU_RUNNER_SRCS += \
		$(wildcard llama/ggml-cuda/fattn*.cu) \
		$(wildcard llama/ggml-cuda/template-instances/fattn-wmma*.cu) \
		$(wildcard llama/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu) \
		$(wildcard llama/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu) \
		$(wildcard llama/ggml-cuda/template-instances/fattn-vec*f16-f16.cu)

GPU_RUNNER_OBJS := $(GPU_RUNNER_SRCS:.cu=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(GPU_RUNNER_OBJS:.c=.$(GPU_RUNNER_NAME).$(OBJ_EXT))
GPU_RUNNER_OBJS := $(addprefix $(BUILD_DIR)/,$(GPU_RUNNER_OBJS:.cpp=.$(GPU_RUNNER_NAME).$(OBJ_EXT)))

$(GPU_RUNNER_NAME): $(BUILD_RUNNERS) 

# Build targets
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cu
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CFLAGS) $(GPU_COMPILER_CUFLAGS) $(GPU_RUNNER_ARCH_FLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.c
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CFLAGS) -o $@ $<
$(BUILD_DIR)/%.$(GPU_RUNNER_NAME).$(OBJ_EXT): %.cpp
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) -c $(GPU_COMPILER_CXXFLAGS) -o $@ $<

$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/llm-server$(EXE_EXT): TARGET_CGO_LDFLAGS = $(CGO_EXTRA_LDFLAGS) -L"$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/"
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/llm-server$(EXE_EXT): $(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT) ./llama/*.go ./main/*.go $(COMMON_SRCS) $(COMMON_HDRS)
	@-mkdir -p $(dir $@)
	GOARCH=amd64 CGO_LDFLAGS="-L/usr/local/cuda-12/lib64 -L/usr/local/cuda-12/lib64/stubs -L./llama/build/linux-amd64/runners/cuda_v12_avx/" go build -buildmode=pie -ldflags="-w -s -X=github.com/ollama/ollama/version.Version=0.5.7-1-g021817e-dirty" -tags=avx,cuda,cuda_v12 -trimpath -o llama/build/linux-amd64/runners/cuda_v12_avx/llm-server ./main
$(RUNNERS_BUILD_DIR)/$(GPU_RUNNER_NAME)$(GPU_RUNNER_EXTRA_VARIANT)/$(SHARED_PREFIX)ggml_$(GPU_RUNNER_NAME).$(SHARED_EXT): $(GPU_RUNNER_OBJS) $(COMMON_HDRS) $(GPU_RUNNER_HDRS)
	@-mkdir -p $(dir $@)
	$(CCACHE) $(GPU_COMPILER) --shared -L$(GPU_LIB_DIR) $(GPU_RUNNER_DRIVER_LIB_LINK) -L${DIST_GPU_RUNNER_DEPS_DIR} $(foreach lib, $(GPU_RUNNER_LIBS_SHORT), -l$(lib)) $(GPU_RUNNER_OBJS) -o $@

clean: 
	rm -f $(GPU_RUNNER_OBJS) $(BUILD_RUNNERS)
	rm -rf $(BUILD_DIR) $(DIST_LIB_DIR) $(OLLAMA_EXE) $(DIST_OLLAMA_EXE)
	go clean -cache

.PHONY: clean $(GPU_RUNNER_NAME)

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'
