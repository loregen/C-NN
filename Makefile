CC = gcc

CFLAGS = -Wall -Wextra -Ofast
DEBUG_CFLAGS = -Wall -Wextra -g

LFLAGS = -lm -lpthread
DEBUG_LFLAGS = -lm -lpthread -g

ifeq ($(shell uname -s),Darwin)
    LFLAGS += -framework Accelerate
    CFLAGS += -DACCELERATE_NEW_LAPACK
    DEBUG_LFLAGS += -framework Accelerate
    DEBUG_CFLAGS += -DACCELERATE_NEW_LAPACK
else
    LFLAGS += -lblas
    DEBUG_LFLAGS += -lblas
endif

OUT_EXT = .out

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

# Common source files
COMMON_SOURCES := net.c array.c layers.c data.c tensor.c
COMMON_OBJECTS := $(COMMON_SOURCES:%.c=$(OBJDIR)/%.o)
COMMON_DEBUG_OBJECTS := $(COMMON_SOURCES:%.c=$(OBJDIR)/%_dbg.o)

# Target for mnistnet executable
mnistnet: $(BINDIR)/mnistnet$(OUT_EXT)

$(BINDIR)/mnistnet$(OUT_EXT): $(SRCDIR)/MNIST_DENSE.c $(COMMON_OBJECTS)
	$(CC) $^ $(LFLAGS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Debug target for mnistnet
mnistnet_debug: $(BINDIR)/mnistnet_dbg$(OUT_EXT)

$(BINDIR)/mnistnet_dbg$(OUT_EXT): $(SRCDIR)/MNIST_DENSE.c $(COMMON_DEBUG_OBJECTS)
	$(CC) $^ $(DEBUG_LFLAGS) -o $@

$(OBJDIR)/%_dbg.o: $(SRCDIR)/%.c
	$(CC) $(DEBUG_CFLAGS) -c $< -o $@

clean:
	rm -f $(BINDIR)/mnistnet$(OUT_EXT)
	rm -f $(BINDIR)/mnistnet_dbg$(OUT_EXT)
	rm -f $(COMMON_OBJECTS) $(COMMON_DEBUG_OBJECTS)

depend: $(COMMON_SOURCES)
	makedepend $(INCLUDES) $^