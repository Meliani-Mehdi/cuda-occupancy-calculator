#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include "cuda_arch.h"

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(path) _mkdir(path)
#else
  #include <unistd.h>
  #define MAKE_DIR(path) mkdir(path, 0755)
#endif

Gpu_t new_GPU(
  char* name,
  unsigned long global_mem_size_in_bytes,
  unsigned int shared_mem_size_in_bytes_per_SM,
  unsigned int number_of_registers_per_SM,
  unsigned short maximum_number_of_warps_per_SM,
  unsigned short maximum_number_of_blocks_per_SM,
  unsigned short number_of_SMs
){

  Gpu_t gpu = {
    .name = strdup(name),
    .global_mem_size_in_bytes = global_mem_size_in_bytes,
    .shared_mem_size_in_bytes_per_SM = shared_mem_size_in_bytes_per_SM,
    .number_of_registers_per_SM = number_of_registers_per_SM,
    .maximum_number_of_warps_per_SM = maximum_number_of_warps_per_SM,
    .maximum_number_of_blocks_per_SM = maximum_number_of_blocks_per_SM,
    .number_of_SMs = number_of_SMs,
  };

  gpu.list_of_SMs = malloc(sizeof(struct SM) * gpu.number_of_SMs);
  if (!gpu.list_of_SMs) {
    perror("Failed to allocate SMs");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < gpu.number_of_SMs; i++) {
    gpu.list_of_SMs[i].number_of_blocks = 0;
    gpu.list_of_SMs[i].list_of_blocks = calloc(gpu.maximum_number_of_blocks_per_SM, sizeof(Block_t));
    if (!gpu.list_of_SMs[i].list_of_blocks) {
      perror("Failed to allocate Block");
      exit(EXIT_FAILURE);
    }
  }

  return gpu;
}

void free_GPU(Gpu_t* gpu){
  for (int i = 0; i < gpu->number_of_SMs; i++) {
    free(gpu->list_of_SMs[i].list_of_blocks);
  }
  free(gpu->list_of_SMs);
}

void clear_kernel_blocks(Gpu_t* gpu, Kernel_t* kernel) {
  if (!gpu || !kernel) {
    fprintf(stderr, "Error: GPU or Kernel pointer is NULL.\n");
    return;
  }

  for (int i = 0; i < gpu->number_of_SMs; i++) {
    SM_t* sm = &gpu->list_of_SMs[i];
    int write_index = 0;

    for (int j = 0; j < sm->number_of_blocks; j++) {
      Block_t* blk = &sm->list_of_blocks[j];

      if (!strcmp(blk->kernel_name, kernel->name)) {
        continue;
      }

      if (write_index != j) {
        sm->list_of_blocks[write_index] = sm->list_of_blocks[j];
      }
      write_index++;
    }

    sm->number_of_blocks = write_index;
  }
}

void print_GPU_info(Gpu_t* gpu) {
  if (!gpu) {
    printf("GPU pointer is NULL.\n");
    return;
  }

  printf("============================================================\n");
  printf(" GPU INFORMATION REPORT\n");
  printf("============================================================\n");
  printf("Name: %s\n", gpu->name);
  printf("------------------------------------------------------------\n");
  printf("Global Memory Size:             %lu bytes (%.2f MB)\n",
         gpu->global_mem_size_in_bytes,
         gpu->global_mem_size_in_bytes / (1024.0 * 1024.0));
  printf("Shared Memory per SM:           %u bytes (%.2f KB)\n",
         gpu->shared_mem_size_in_bytes_per_SM,
         gpu->shared_mem_size_in_bytes_per_SM / 1024.0);
  printf("Registers per SM:               %u\n", gpu->number_of_registers_per_SM);
  printf("Max Warps per SM:               %hu\n", gpu->maximum_number_of_warps_per_SM);
  printf("Max Blocks per SM:              %hu\n", gpu->maximum_number_of_blocks_per_SM);
  printf("Number of SMs:                  %hu\n", gpu->number_of_SMs);
  printf("------------------------------------------------------------\n");

  if (!gpu->list_of_SMs) {
    printf("No SMs available (list_of_SMs is NULL)\n");
    return;
  }

  // Iterate over SMs
  for (unsigned short sm_idx = 0; sm_idx < gpu->number_of_SMs; ++sm_idx) {
    SM_t* sm = &gpu->list_of_SMs[sm_idx];
    if (!sm) continue;

    printf("\n[SM %hu]\n", sm_idx);
    printf("------------------------------------------------------------\n");
    printf("Number of Active Blocks: %u / %hu (%.2f%% utilization)\n",
           sm->number_of_blocks,
           gpu->maximum_number_of_blocks_per_SM,
           100.0 * sm->number_of_blocks / gpu->maximum_number_of_blocks_per_SM);

    unsigned int total_shared_used = 0;
    unsigned int total_registers_used = 0;
    unsigned int total_threads = 0;

    if (sm->list_of_blocks) {
      for (unsigned int blk_idx = 0; blk_idx < sm->number_of_blocks; ++blk_idx) {
        Block_t* block = &sm->list_of_blocks[blk_idx];
        total_shared_used += block->shared_mem_used_in_bytes;
        total_registers_used += block->number_of_registers_used_per_thread * block->number_of_thread;
        total_threads += block->number_of_thread;
      }
    }

    int max_threads = gpu->maximum_number_of_warps_per_SM * 32;

    printf("Total Threads:                  %u / %u (%.2f%%)\n", 
           total_threads,
           max_threads,
           100.0 * total_threads / max_threads);
    printf("Total Shared Memory Used:       %u / %u bytes (%.2f%%)\n",
           total_shared_used,
           gpu->shared_mem_size_in_bytes_per_SM,
           100.0 * total_shared_used / gpu->shared_mem_size_in_bytes_per_SM);
    printf("Total Registers Used:           %u / %u (%.2f%%)\n",
           total_registers_used,
           gpu->number_of_registers_per_SM,
           100.0 * total_registers_used / gpu->number_of_registers_per_SM);

    // Add Occupancy Info Here
    print_occupancy_of_SM(gpu, sm_idx);

    printf("\n  BLOCKS IN SM %hu:\n", sm_idx);
    printf("  ----------------------------------------------------------\n");
    if (!sm->list_of_blocks) {
      printf("  No blocks in this SM.\n");
    } else {
      for (unsigned int blk_idx = 0; blk_idx < sm->number_of_blocks; ++blk_idx) {
        Block_t* block = &sm->list_of_blocks[blk_idx];
        printf("  [Block %u]\n", blk_idx);
        printf("    Kernel Name:                %s\n", block->kernel_name);
        printf("    Threads:                    %u\n", block->number_of_thread);
        printf("    Shared Memory Used:         %u bytes\n", block->shared_mem_used_in_bytes);
        printf("    Registers per Thread:       %u\n", block->number_of_registers_used_per_thread);
        printf("    Total Registers Used:       %u\n",
               block->number_of_thread * block->number_of_registers_used_per_thread);
        printf("  ----------------------------------------------------------\n");
      }
    }
  }

  printf("\n============================================================\n");
  printf(" END OF GPU REPORT\n");
  printf("============================================================\n\n");
}

void export_GPU_to_HTML(Gpu_t* gpu) {
  if (!gpu) {
    fprintf(stderr, "Error: GPU pointer is NULL.\n");
    return;
  }

  // Ensure "results" directory exists
  struct stat st = {0};
  if (stat("results", &st) == -1) {
    if (MAKE_DIR("results") != 0 && errno != EEXIST) {
      fprintf(stderr, "Error creating results/ folder: %s\n", strerror(errno));
      return;
    }
  }

  // Build safe file name (replace spaces)
  char safe_name[256];
  snprintf(safe_name, sizeof(safe_name), "%s", gpu->name);
  for (char* p = safe_name; *p; p++) {
    if (*p == ' ') *p = '_';
  }

  char filepath[512];
  snprintf(filepath, sizeof(filepath), "results/%s.html", safe_name);

  FILE* f = fopen(filepath, "w");
  if (!f) {
    fprintf(stderr, "Error: could not open file %s for writing.\n", filepath);
    return;
  }

  // HTML Header
  fprintf(f,
          "<!DOCTYPE html>\n"
          "<html lang='en'>\n"
          "<head>\n"
          "  <meta charset='UTF-8'>\n"
          "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
          "  <title>GPU: %s</title>\n"
          "  <link rel='stylesheet' href='../gpu_style.css'>\n"
          "</head>\n"
          "<body>\n"
          "  <h1>GPU: %s</h1>\n"
          "  <div class='gpu-container'>\n"
          "    <div class='gpu-header'>Global Memory: %.2f GB | SMs: %hu | Shared Mem/SM: %.2f KB | Registers/SM: %u</div>\n"
          "    <div class='sm-grid'>\n",
          gpu->name,
          gpu->name,
          (double)gpu->global_mem_size_in_bytes / (1024.0 * 1024.0 * 1024.0),
          gpu->number_of_SMs,
          (double)gpu->shared_mem_size_in_bytes_per_SM / 1024.0,
          gpu->number_of_registers_per_SM
          );

  // Loop through SMs
  for (int i = 0; i < gpu->number_of_SMs; i++) {
    SM_t* sm = &gpu->list_of_SMs[i];
    double occ = 0.0;
    if (gpu->maximum_number_of_warps_per_SM > 0)
      occ = calculate_occupancy_of_SM(gpu, i) * 100.0;

    unsigned int total_shared = 0;
    unsigned int total_regs = 0;
    for (int b = 0; b < sm->number_of_blocks; b++) {
      Block_t* blk = &sm->list_of_blocks[b];
      total_shared += blk->shared_mem_used_in_bytes;
      total_regs += blk->number_of_registers_used_per_thread * blk->number_of_thread;
    }

    fprintf(f,
            "      <div class='sm'>\n"
            "        <h3 class='sm_text'>SM #%d</h3>\n"
            "        <div class='tooltip_sm'>\n"
            "          Occupancy: %.2f%%<br>\n"
            "          Blocks: %hu / %hu<br>\n"
            "          Shared Mem Used: %.2f%%<br> %.1f KB / %.1f KB<br>\n"
            "          Registers Used: %.2f%%<br> %.1fK / %.1fK\n"
            "        </div>\n"
            "        <div class='block-container'>\n",
            i,
            occ,
            sm->number_of_blocks,
            gpu->maximum_number_of_blocks_per_SM,
            (double)100.0 * total_shared / gpu->shared_mem_size_in_bytes_per_SM,
            (double)total_shared / 1024.0,
            (double)gpu->shared_mem_size_in_bytes_per_SM / 1024.0,
            (double)100.0 * total_regs / gpu->number_of_registers_per_SM,
            (double)total_regs / 1000.0,
            (double)gpu->number_of_registers_per_SM / 1000.0
            );

    // Loop through Blocks
    for (int b = 0; b < sm->number_of_blocks; b++) {
      Block_t* blk = &sm->list_of_blocks[b];
      fprintf(f,
              "          <div class='block'>\n"
              "            <div class='tooltip'>\n"
              "              Kernel: %s<br>\n"
              "              Threads: %u<br>\n"
              "              Shared Mem: %.1f KB<br>\n"
              "              Regs/thread: %u\n"
              "            </div>\n"
              "          </div>\n",
              blk->kernel_name,
              blk->number_of_thread,
              (double)blk->shared_mem_used_in_bytes / 1024.0,
              blk->number_of_registers_used_per_thread
              );
    }

    fprintf(f, "        </div>\n      </div>\n");
  }

  fprintf(f,
          "    </div>\n"
          "  </div>\n"
          "</body>\n</html>\n"
          );

  fclose(f);
  printf("HTML visualization generated: %s\n", filepath);
}

void print_occupancy_of_all_SMs(Gpu_t* gpu){
  for (int i=0; i < gpu->number_of_SMs; i++) {
    print_occupancy_of_SM(gpu, i);
  }
  printf("\n");
}

void print_occupancy_of_SM(Gpu_t* gpu, int SM_pos){
  if(!gpu){
    printf("GPU pointer is NULL.\n");
    return;
  }
  if(SM_pos < 0 || SM_pos >= gpu->number_of_SMs){
    printf("Invalid SM position: %d\n", SM_pos);
    return;
  }

  double occupancy = calculate_occupancy_of_SM(gpu, SM_pos);
  printf("\n  >>> Occupancy of SM %d: %.2f%% <<<\n", SM_pos, occupancy * 100.0);
}

double calculate_occupancy_of_SM(Gpu_t* gpu, int SM_pos) {
  if (!gpu || SM_pos < 0 || SM_pos >= gpu->number_of_SMs) return 0.0;
  SM_t* sm = &(gpu->list_of_SMs[SM_pos]);
  if (!sm || sm->number_of_blocks == 0) return 0.0;

  unsigned int total_warps = 0;
  unsigned int total_registers = 0;
  unsigned int total_shared = 0;

  for (int i = 0; i < sm->number_of_blocks; i++) {
    Block_t* b = &sm->list_of_blocks[i];
    unsigned int warps = (b->number_of_thread + 31) / 32;
    total_warps += warps;
    total_registers += b->number_of_registers_used_per_thread * b->number_of_thread;
    total_shared += b->shared_mem_used_in_bytes;
  }

  double warp_occ = (double)total_warps / gpu->maximum_number_of_warps_per_SM;
  double reg_occ = (double)total_registers / gpu->number_of_registers_per_SM;
  double shm_occ = (double)total_shared / gpu->shared_mem_size_in_bytes_per_SM;
  double blk_occ = (double)sm->number_of_blocks / gpu->maximum_number_of_blocks_per_SM;

  // Effective occupancy limited by any resource that saturates first
  double occupancy = warp_occ;
  if (reg_occ > occupancy) occupancy = reg_occ;
  if (shm_occ > occupancy) occupancy = shm_occ;
  if (blk_occ > occupancy) occupancy = blk_occ;

  // Limit to 1.0 max
  if (occupancy > 1.0) occupancy = 1.0;

  return occupancy;
}

void make_stream_queues(
  Kernel_t* kernel_arr,
  int arr_size,
  StreamQueue_t** out_streams,
  int* out_count
) {
  if (arr_size == 0) {
    *out_streams = NULL;
    *out_count = 0;
    return;
  }

  // Find max stream_id to size lookup table
  unsigned short max_stream_id = 0;
  for (int i = 0; i < arr_size; i++) {
    if (kernel_arr[i].stream_id > max_stream_id) {
      max_stream_id = kernel_arr[i].stream_id;
    }
  }

  int* stream_counts = calloc(max_stream_id + 1, sizeof(int));
  if (!stream_counts) {
    perror("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Count kernels per stream
  for (int i = 0; i < arr_size; i++) {
    stream_counts[kernel_arr[i].stream_id]++;
  }

  // Count how many unique streams
  int stream_count = 0;
  for (int i = 0; i <= max_stream_id; i++) {
    if (stream_counts[i] > 0) {
      stream_count++;
    }
  }

  // Allocate only as many StreamQueue_t as needed
  StreamQueue_t* streams = malloc(sizeof(StreamQueue_t) * stream_count);
  if (!streams) {
    perror("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Mapping ids to a index and initializing queues
  int* id_to_index = malloc(sizeof(int) * (max_stream_id + 1));
  if (!id_to_index) {
    perror("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  int idx = 0;
  for (int i = 0; i <= max_stream_id; i++) {
    if (stream_counts[i] > 0) {
      streams[idx].stream_id = i;
      queue_kernel_init(&streams[idx].queue, stream_counts[i]);
      id_to_index[i] = idx;
      idx++;
    } else {
      id_to_index[i] = -1; // unused stream_id
    }
  }

  // Enqueue kernels into the correct stream’s queue
  for (int i = 0; i < arr_size; i++) {
    int index = id_to_index[kernel_arr[i].stream_id];
    queue_kernel_enqueue(&streams[index].queue, kernel_arr[i]);
  }

  free(stream_counts);
  free(id_to_index);

  *out_streams = streams;
  *out_count = stream_count;
}



// this function needs a whole rewrite or maybe i should delete it
queue_kernel_t* ready_EE_queue(StreamQueue_t* streams, int streams_size, int number_of_kernels) {
  if (streams_size == 0 || number_of_kernels == 0) return NULL;

  queue_kernel_t* EE_queue = malloc(sizeof(queue_kernel_t));
  if (!EE_queue) {
    perror("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  queue_kernel_init(EE_queue, number_of_kernels);

  // Round robin with flush at stream 0
  bool active = true;
  while (active) {
    active = 0;
    for (int i = 0; i < streams_size; i++) {
      if (queue_kernel_empty(&streams[i].queue)) {
        continue;
      }

      if (streams[i].stream_id == 0) {
        // Flush ALL kernels of stream 0
        while (!queue_kernel_empty(&streams[i].queue)) {
          Kernel_t k = queue_kernel_dequeue(&streams[i].queue);
          queue_kernel_enqueue(EE_queue, k);
        }
      } else {
        // Take only one kernel
        Kernel_t k = queue_kernel_dequeue(&streams[i].queue);
        queue_kernel_enqueue(EE_queue, k);
      }

      active = false; // at least one kernel was taken this pass
    }
  }

  return EE_queue;
}

bool canFitBlock(Gpu_t* gpu, int sm_pos, Block_t* block) {
  if (!gpu || !block) return false;
  if (sm_pos < 0 || sm_pos >= gpu->number_of_SMs) return false;

  SM_t* sm = &(gpu->list_of_SMs[sm_pos]);

  if (sm->number_of_blocks >= gpu->maximum_number_of_blocks_per_SM)
    return false;

  // Compute total used resources
  unsigned long used_shared = 0, used_regs = 0, used_warps = 0;

  for (unsigned short i = 0; i < sm->number_of_blocks; ++i) {
    Block_t* b = &sm->list_of_blocks[i];
    used_warps += (unsigned long)((b->number_of_thread + 31) / 32 );
    used_shared += b->shared_mem_used_in_bytes;
    used_regs   += (unsigned long)b->number_of_registers_used_per_thread * 
      (unsigned long)b->number_of_thread;
  }

  unsigned long new_warps = used_warps + (unsigned long)((block->number_of_thread + 31) / 32 );
  unsigned long new_shared = used_shared + (unsigned long)block->shared_mem_used_in_bytes;
  unsigned long new_regs   = used_regs + 
    ((unsigned long)block->number_of_registers_used_per_thread * 
    (unsigned long)block->number_of_thread);

  // Compare against SM limits
  if (new_warps > gpu->maximum_number_of_warps_per_SM)
    return false;

  if (new_shared > gpu->shared_mem_size_in_bytes_per_SM)
    return false;

  if (new_regs > gpu->number_of_registers_per_SM)
    return false;

  return true;
}

void launch_one_kernel(Gpu_t* gpu, Kernel_t* kernel){
  Block_t* blocks = calloc(kernel->number_of_blocks, sizeof(Block_t));
  if (!blocks) {
    perror("Failed to allocate blocks");
    return;
  }

  int i;

  for (i = 0; i < kernel->number_of_blocks; i++) {
    blocks[i].kernel_name = kernel->name;
    blocks[i].number_of_thread = kernel->threads_per_block;
    blocks[i].shared_mem_used_in_bytes = kernel->shared_mem_used_in_bytes_per_block;
    blocks[i].number_of_registers_used_per_thread = kernel->registers_per_thread;
  }
  int count = 0;
  i = 0;
  bool even = true, retry_flag = false;
  while (count < kernel->number_of_blocks) {
    // if we checked all even SMs we go to odd SMs and vice versa
    if(i >= gpu->number_of_SMs){
      i = (even)? 1 : 0;
      even = !even;

      if(retry_flag){
        // print a better error later, TO DO, DONT FORGET.
        printf("%d of blocks of kernel %s did not fit in the GPU %s\n", kernel->number_of_blocks - count, kernel->name, gpu->name);
        free(blocks);
        return;
      }
      else{
        retry_flag = true;
      } 
    }

    if(canFitBlock(gpu, i, &blocks[count])){
      retry_flag = false;
      gpu->list_of_SMs[i].list_of_blocks[gpu->list_of_SMs[i].number_of_blocks++] = blocks[count++];
    }

    i += 2;
  }

  free(blocks);
  printf("all blocks of kernel %s run succesfuly!\n", kernel->name);

}

//  work in progress
void launch_kernels(Gpu_t* gpu, Kernel_t* kernel_arr, int arr_size){
  StreamQueue_t* streams;
  int stream_count;
  make_stream_queues(kernel_arr, arr_size, &streams, &stream_count);
  queue_kernel_t* EE_queue = ready_EE_queue(streams, stream_count, arr_size);

}
