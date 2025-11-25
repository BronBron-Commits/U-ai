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
      src/mha.c \
      src/init.c \
      src/tokenizer.c \
      src/embedding.c

OUT = build/uai

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)
