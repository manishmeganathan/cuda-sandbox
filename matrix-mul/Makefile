TARGET = matrix_mul

build:
	@nvcc $(TARGET).cu -o $(TARGET) -Wno-deprecated-gpu-targets

--nsys-prof: build
	@nsys profile -t cuda --stats=true ./$(TARGET)

run: build
	@./$(TARGET)

clean:
	@rm -f $(TARGET)
	@rm -f report*

profile: --nsys-prof clean

all: run clean