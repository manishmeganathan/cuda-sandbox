TARGET ?= torch_softmax
BUILD_DIR ?= build
CUDA_ARCHS  ?= 86;90
TORCH_ARCHS  = "8.6"
CONFIG    ?= Release

LIBTORCH = /workspace/libtorch

.PHONY: build run profile clean all

# CMake configure (runs if build has been cleaned)
$(BUILD_DIR)/CMakeCache.txt:
	cmake -B $(BUILD_DIR) -S . \
	  -DCMAKE_BUILD_TYPE=$(CONFIG) \
	  -DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHS)" \
	  -DCMAKE_PREFIX_PATH="$(LIBTORCH)" \
	  -DTORCH_CUDA_ARCH_LIST=$(TORCH_ARCHS)

build: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) -j

--nsys-prof: build
	@nsys profile -t cuda --stats=true ./$(TARGET)

run: build
	@./$(BUILD_DIR)/$(TARGET)

clean:
	@rm -f report*
	@rm -rf $(BUILD_DIR)

profile: --nsys-prof clean

all: run clean
