TARGET = vector_add

build:
	@nvcc $(TARGET).cu -o $(TARGET) -Wno-deprecated-gpu-targets

run: build
	@./$(TARGET)

clean:
	@rm -f $(TARGET)

all: run clean