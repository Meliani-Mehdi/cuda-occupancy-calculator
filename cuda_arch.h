#ifndef CUDA_ARCH_H
#define CUDA_ARCH_H

#include <stdio.h>
#include <stdlib.h>
#include "queue.h"

// ================= Type Declaration ==================

typedef struct KERNEL {
  char* name;

  unsigned int number_of_blocks;
  unsigned int threads_per_block;

  unsigned int shared_mem_used_in_bytes_per_block;
  unsigned int registers_per_thread;

  unsigned short stream_id;
} Kernel_t;

QUEUE_DEFINE(Kernel_t, kernel)

typedef struct STREAM_QUEUE{
  unsigned short stream_id;
  queue_kernel_t queue;
} StreamQueue_t;

typedef struct BLOCK {
  char* kernel_name;
  unsigned int number_of_thread;
  unsigned int shared_mem_used_in_bytes;
  unsigned int number_of_registers_used_per_thread;
} Block_t;

typedef struct SM {
  unsigned short number_of_blocks;
  Block_t* list_of_blocks;
} SM_t;

typedef struct GPU {
  const char* name;

  const unsigned long global_mem_size_in_bytes;
  const unsigned int shared_mem_size_in_bytes_per_SM;
  const unsigned int number_of_registers_per_SM;
  const unsigned short maximum_number_of_warps_per_SM;
  const unsigned short maximum_number_of_blocks_per_SM;

  const unsigned short number_of_SMs;
  SM_t* list_of_SMs;
} Gpu_t;

// ================= Function Declarations ==================

Gpu_t new_GPU(
  const char* name,
  const unsigned long global_mem_size_in_bytes,
  const unsigned int shared_mem_size_in_bytes_per_SM,
  const unsigned int number_of_registers_per_SM,
  const unsigned short maximum_number_of_warps_per_SM,
  const unsigned short maximum_number_of_blocks_per_SM,
  const unsigned short number_of_SMs
);

void free_GPU(Gpu_t* gpu);

void clear_kernel_blocks(Gpu_t* gpu, Kernel_t* kernel);

void print_GPU_info(Gpu_t* gpu);

void export_GPU_to_HTML(Gpu_t* gpu);

bool canFitBlock(Gpu_t* gpu, int sm_pos, Block_t* block);

void launch_one_kernel(Gpu_t* gpu, Kernel_t* kernel);

void launch_kernels(Gpu_t* gpu, Kernel_t* kernel_arr, int arr_size);

double calculate_occupancy_of_SM(Gpu_t* gpu, int SM_pos);

void print_occupancy_of_SM(Gpu_t* gpu, int SM_pos);

void print_occupancy_of_all_SMs(Gpu_t* gpu);

void make_stream_queues(
  Kernel_t* kernel_arr,
  int arr_size,
  StreamQueue_t** out_streams,
  int* out_count
);

queue_kernel_t* ready_EE_queue(StreamQueue_t* streams, int streams_size, int number_of_kernels);

#endif // CUDA_ARCH_H
