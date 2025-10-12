TARGET = matrix

MNIST_SRC = https://github.com/phoebetronic/mnist.git
MNIST_TEST_FILE = mnist_test.csv
MNIST_TRAIN_FILE = mnist_train.csv

build:
	@nvcc $(TARGET).cu -o $(TARGET) -Wno-deprecated-gpu-targets

--nsys-prof: build
	@nsys profile -t cuda --stats=true ./$(TARGET)

run: build
	@./$(TARGET)

download:
	git clone $(MNIST_SRC)

	mv ./mnist/$(MNIST_TEST_FILE).zip ./$(MNIST_TEST_FILE).zip
	mv ./mnist/$(MNIST_TRAIN_FILE).zip ./$(MNIST_TRAIN_FILE).zip

	rm -rf ./mnist

	unzip $(MNIST_TEST_FILE).zip
	unzip $(MNIST_TRAIN_FILE).zip

	rm -rf __*
	rm -rf *.zip

clean:
	@rm -f $(TARGET)
	@rm -f report*

profile: --nsys-prof clean

all: run clean