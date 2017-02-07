
CC = gcc

CFLAGS = -Wall -Werror --pedantic -Iinclude -lm -g

SRCS=$(wildcard src/*.c)
OBJS=$(SRCS:.c=.o)

EXEC=nnet

all: $(OBJS)
	$(CC) -o $(EXEC) $^ $(CFLAGS)

clean:
	rm -f $(OBJS)

fclean:
	rm -f $(OBJS) $(EXEC)

re:
	make fclean all
