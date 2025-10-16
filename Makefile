# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g

# Source and build directories
SRC_DIR = code
BUILD_DIR = build

# Source files
SRCS = $(SRC_DIR)/GPU_sim.c $(SRC_DIR)/cuda_arch.c $(SRC_DIR)/cJSON.c

# Object files go into build/
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Output executable
TARGET = GPU_sim

# Default rule
all: $(TARGET)

# Link object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Rule to build object files into build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean compiled files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Remove result files
clear:
	rm -rf ./results
	mkdir results

# Force rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)
