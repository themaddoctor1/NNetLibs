
CC = gcc

CFLAGS = -Wall -Werror --pedantic -Iinclude -lm -g

SRCS=$(wildcard src/*.c)
OBJS=$(SRCS:.c=.o)

LIB=libnnet.a

all: $(OBJS)
	ar -rcs $(LIB) $(OBJS)

clean:
	rm -f $(OBJS)

fclean:
	rm -f $(OBJS) $(LIB)

re:
	make fclean all
