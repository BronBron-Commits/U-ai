CC = clang
CFLAGS = -O3 -Wall -Iinclude
LDFLAGS = -lm

SRC = src/main.c \
      src/tensor.c \
      src/linear.c \
      src/activation.c \
      src/mlp.c \
      src/layernorm.c \
      src/softmax.c \
      src/attention.c \
      src/init.c \
      src/mha.c

OUT = build/uai

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)
