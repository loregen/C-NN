CC = gcc

CFLAGS = -Wall -Wextra -Ofast
DEBUG_CFLAGS = -Wall -Wextra -g

LFLAGS = -lm -lpthread 
DEBUG_LFLAGS = -lm -lpthread -g

ifeq ($(shell uname -s),Darwin)
    LFLAGS += -framework Accelerate
    CFLAGS += -DACCELERATE_NEW_LAPACK
    DEBUG_LFLAGS += -framework Accelerate
else
    LFLAGS += -lblas
    DEBUG_LFLAGS += -lblas
endif

MAIN = mnistnet

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.c)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEBUGOBJECTS = $(OBJECTS:.o=_dbg.o) # Add this line

$(BINDIR)/$(MAIN): $(OBJECTS)
	$(CC) $(OBJECTS) $(LFLAGS) -o $@

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

debug: $(DEBUGOBJECTS)
	$(CC) $(DEBUGOBJECTS) $(DEBUG_LFLAGS) -o $(BINDIR)/$(MAIN)_dbg

$(DEBUGOBJECTS): $(OBJDIR)/%_dbg.o : $(SRCDIR)/%.c 
	$(CC) $(DEBUG_CFLAGS) -c $< -o $@

clean:
	rm -f $(MAIN) $(OBJECTS) $(DEBUGOBJECTS) 

depend: $(SRCS)
	makedepend $(INCLUDES) $^
