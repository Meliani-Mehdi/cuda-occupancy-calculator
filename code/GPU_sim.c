#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cuda_arch.h"
#include "cJSON.h"

#define CONFIG_FILE "config.json"

void load_config(const char *filename, Gpu_t **gpus, int *gpu_count, Kernel_t **kernels, int *kernel_count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open config file");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    rewind(fp);

    char *data = malloc(len + 1);
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        exit(1);
    }

    fread(data, 1, len, fp);
    data[len] = '\0';
    fclose(fp);

    cJSON *root = cJSON_Parse(data);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(data);
        exit(1);
    }

    // --- GPUs ---
    cJSON *gpu_array = cJSON_GetObjectItem(root, "gpus");
    if (!cJSON_IsArray(gpu_array)) {
        fprintf(stderr, "Error: 'gpus' field missing or not an array\n");
        cJSON_Delete(root);
        free(data);
        exit(1);
    }

    *gpu_count = cJSON_GetArraySize(gpu_array);
    *gpus = malloc(sizeof(Gpu_t) * (*gpu_count));
    if (!*gpus) {
        fprintf(stderr, "Memory allocation failed for GPUs\n");
        cJSON_Delete(root);
        free(data);
        exit(1);
    }

    for (int i = 0; i < *gpu_count; i++) {
        cJSON *gpu = cJSON_GetArrayItem(gpu_array, i);
        if (!cJSON_IsObject(gpu)) {
            fprintf(stderr, "Warning: GPU[%d] is not a valid object, skipping\n", i);
            continue;
        }

        cJSON *j_name = cJSON_GetObjectItem(gpu, "name");
        cJSON *j_mem = cJSON_GetObjectItem(gpu, "memory_bytes");
        cJSON *j_shared = cJSON_GetObjectItem(gpu, "shared_mem_per_sm");
        cJSON *j_regs = cJSON_GetObjectItem(gpu, "registers_per_sm");
        cJSON *j_warps = cJSON_GetObjectItem(gpu, "max_warps_per_sm");
        cJSON *j_blocks = cJSON_GetObjectItem(gpu, "max_blocks_per_sm");
        cJSON *j_sms = cJSON_GetObjectItem(gpu, "num_sms");

        if (!cJSON_IsString(j_name) || !cJSON_IsNumber(j_mem) || !cJSON_IsNumber(j_shared) ||
            !cJSON_IsNumber(j_regs) || !cJSON_IsNumber(j_warps) ||
            !cJSON_IsNumber(j_blocks) || !cJSON_IsNumber(j_sms)) {
            fprintf(stderr, "Warning: GPU[%d] missing one or more fields, skipping\n", i);
            continue;
        }

        (*gpus)[i] = new_GPU(
            j_name->valuestring,
            (unsigned long) j_mem->valuedouble,
            j_shared->valueint,
            j_regs->valueint,
            j_warps->valueint,
            j_blocks->valueint,
            j_sms->valueint
        );
    }

    // --- Kernels ---
    cJSON *kernel_array = cJSON_GetObjectItem(root, "kernels");
    if (!cJSON_IsArray(kernel_array)) {
        fprintf(stderr, "Error: 'kernels' field missing or not an array\n");
        cJSON_Delete(root);
        free(data);
        exit(1);
    }

    *kernel_count = cJSON_GetArraySize(kernel_array);
    *kernels = malloc(sizeof(Kernel_t) * (*kernel_count));
    if (!*kernels) {
        fprintf(stderr, "Memory allocation failed for kernels\n");
        cJSON_Delete(root);
        free(data);
        exit(1);
    }

    for (int i = 0; i < *kernel_count; i++) {
        cJSON *k = cJSON_GetArrayItem(kernel_array, i);
        if (!cJSON_IsObject(k)) {
            fprintf(stderr, "Warning: Kernel[%d] is not a valid object, skipping\n", i);
            continue;
        }

        cJSON *j_name = cJSON_GetObjectItem(k, "name");
        cJSON *j_blocks = cJSON_GetObjectItem(k, "number_of_blocks");
        cJSON *j_threads = cJSON_GetObjectItem(k, "threads_per_block");
        cJSON *j_shared = cJSON_GetObjectItem(k, "shared_mem_used_in_bytes_per_block");
        cJSON *j_regs = cJSON_GetObjectItem(k, "registers_per_thread");
        cJSON *j_stream = cJSON_GetObjectItem(k, "stream_id");

        if (!cJSON_IsString(j_name) || !cJSON_IsNumber(j_blocks) ||
            !cJSON_IsNumber(j_threads) || !cJSON_IsNumber(j_shared) ||
            !cJSON_IsNumber(j_regs) || !cJSON_IsNumber(j_stream)) {
            fprintf(stderr, "Warning: Kernel[%d] missing one or more fields, skipping\n", i);
            continue;
        }

        (*kernels)[i].name = strdup(j_name->valuestring);
        (*kernels)[i].number_of_blocks = j_blocks->valueint;
        (*kernels)[i].threads_per_block = j_threads->valueint;
        (*kernels)[i].shared_mem_used_in_bytes_per_block = j_shared->valueint;
        (*kernels)[i].registers_per_thread = j_regs->valueint;
        (*kernels)[i].stream_id = j_stream->valueint;
    }

    cJSON_Delete(root);
    free(data);
}

int main() {
  Gpu_t *gpus = NULL;
  Kernel_t *kernels = NULL;
  int gpu_count = 0, kernel_count = 0;

  load_config(CONFIG_FILE, &gpus, &gpu_count, &kernels, &kernel_count);

  char dummy;
  for (int g = 0; g < gpu_count; g++) {
    printf("\n==============================\n");
    printf("Launching kernels on %s\n", gpus[g].name);
    printf("==============================\n");

    for (int k = 0; k < kernel_count; k++) {
      launch_one_kernel(&gpus[g], &kernels[k]);
    }

    printf("\nPress ENTER to display info for %s...", gpus[g].name);
    fflush(stdout);
    while ((dummy = getchar()) != '\n' && dummy != EOF);

    print_GPU_info(&gpus[g]);
    export_GPU_to_HTML(&gpus[g]);
    free_GPU(&gpus[g]);
  }

  free(gpus);
  free(kernels);
  return 0;
}
