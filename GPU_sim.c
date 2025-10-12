#include <stdio.h>
#include <stdbool.h>
#include "cuda_arch.h"

int main() {

  // ------------------------------------------------------------
  // Define 3 GPUs with different capabilities
  // ------------------------------------------------------------
  Gpu_t gpu1 = new_GPU("MiniGPU",
                       4UL * 1024 * 1024 * 1024,  // 4 GB
                       48 * 1024,                 // 48 KB shared mem
                       65536,                     // 65k registers
                       64,                        // 64 warps per SM
                       8,                         // 8 blocks per SM
                       4);                        // 4 SMs

  Gpu_t gpu2 = new_GPU("MidGPU",
                       8UL * 1024 * 1024 * 1024,  // 8 GB
                       96 * 1024,                 // 96 KB shared mem
                       131072,                    // 128k registers
                       64,                        // 64 warps per SM
                       16,                        // 16 blocks per SM
                       8);                        // 8 SMs

  Gpu_t gpu3 = new_GPU("MaxGPU",
                       16UL * 1024 * 1024 * 1024, // 16 GB
                       128 * 1024,                // 128 KB shared mem
                       256000,                    // 256k registers
                       64,                        // 64 warps per SM
                       32,                        // 32 blocks per SM
                       16);                       // 16 SMs

  // ------------------------------------------------------------
  // Define 4 kernels with different resource profiles
  // ------------------------------------------------------------
  Kernel_t kernels[4] = {
    {
      .name = "K1_light",
      .number_of_blocks = 8,
      .threads_per_block = 128,
      .shared_mem_used_in_bytes_per_block = 1024,
      .registers_per_thread = 32,
      .stream_id = 1
    },
    {
      .name = "K2_register_heavy",
      .number_of_blocks = 16,
      .threads_per_block = 256,
      .shared_mem_used_in_bytes_per_block = 512,
      .registers_per_thread = 256,
      .stream_id = 1
    },
    {
      .name = "K3_sharedmem_heavy",
      .number_of_blocks = 8,
      .threads_per_block = 256,
      .shared_mem_used_in_bytes_per_block = 48 * 1024, // 48 KB
      .registers_per_thread = 64,
      .stream_id = 1
    },
    {
      .name = "K4_large_threads",
      .number_of_blocks = 32,
      .threads_per_block = 2048,
      .shared_mem_used_in_bytes_per_block = 2048,
      .registers_per_thread = 32,
      .stream_id = 1
    }
  };

  // ------------------------------------------------------------
  // Launch all kernels on each GPU
  // ------------------------------------------------------------
  Gpu_t* gpus[3] = { &gpu1, &gpu2, &gpu3 };
  char dummy;

  for (int g = 0; g < 3; g++) {
    printf("\n==============================\n");
    printf("Launching kernels on %s\n", gpus[g]->name);
    printf("==============================\n");

    for (int k = 0; k < 4; k++) {
      launch_one_kernel(gpus[g], &kernels[k]);
    }

    printf("\nPress ENTER to display info for %s...", gpus[g]->name);
    fflush(stdout);
    // Wait for ENTER
    while ((dummy = getchar()) != '\n' && dummy != EOF);

    print_GPU_info(gpus[g]);
  }

  export_GPU_to_HTML(&gpu1);
  export_GPU_to_HTML(&gpu2);
  export_GPU_to_HTML(&gpu3);
  // ------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------
  free_GPU(&gpu1);
  free_GPU(&gpu2);
  free_GPU(&gpu3);

  return 0;
}
