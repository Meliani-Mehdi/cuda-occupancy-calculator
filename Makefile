# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g

# Source files
SRCS = GPU_sim.c cuda_arch.c cJSON.c

# Object files
OBJS = $(SRCS:.c=.o)

# Output executable
TARGET = GPU_sim

# Default rule
all: $(TARGET)

# Link object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Compile each .c file into .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Remove compiled files
clean:
	rm -f $(OBJS) $(TARGET)

# Remove result files
clear:
	rm -rf ./results
	mkdir results

# Optional: force rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)
