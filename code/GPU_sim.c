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
    fread(data, 1, len, fp);
    data[len] = '\0';
    fclose(fp);

    cJSON *root = cJSON_Parse(data);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(data);
        exit(1);
    }

    cJSON *gpu_array = cJSON_GetObjectItem(root, "gpus");
    *gpu_count = cJSON_GetArraySize(gpu_array);
    *gpus = malloc(sizeof(Gpu_t) * (*gpu_count));

    for (int i = 0; i < *gpu_count; i++) {
        cJSON *gpu = cJSON_GetArrayItem(gpu_array, i);
        char *name = cJSON_GetObjectItem(gpu, "name")->valuestring;
        unsigned long mem = cJSON_GetObjectItem(gpu, "memory_bytes")->valueint;
        int shared_mem = cJSON_GetObjectItem(gpu, "shared_mem_per_sm")->valueint;
        int regs = cJSON_GetObjectItem(gpu, "registers_per_sm")->valueint;
        int warps = cJSON_GetObjectItem(gpu, "max_warps_per_sm")->valueint;
        int blocks = cJSON_GetObjectItem(gpu, "max_blocks_per_sm")->valueint;
        int sms = cJSON_GetObjectItem(gpu, "num_sms")->valueint;

        (*gpus)[i] = new_GPU(name, mem, shared_mem, regs, warps, blocks, sms);
    }

    cJSON *kernel_array = cJSON_GetObjectItem(root, "kernels");
    *kernel_count = cJSON_GetArraySize(kernel_array);
    *kernels = malloc(sizeof(Kernel_t) * (*kernel_count));

    for (int i = 0; i < *kernel_count; i++) {
        cJSON *k = cJSON_GetArrayItem(kernel_array, i);
        (*kernels)[i].name = strdup(cJSON_GetObjectItem(k, "name")->valuestring);
        (*kernels)[i].number_of_blocks = cJSON_GetObjectItem(k, "number_of_blocks")->valueint;
        (*kernels)[i].threads_per_block = cJSON_GetObjectItem(k, "threads_per_block")->valueint;
        (*kernels)[i].shared_mem_used_in_bytes_per_block = cJSON_GetObjectItem(k, "shared_mem_used_in_bytes_per_block")->valueint;
        (*kernels)[i].registers_per_thread = cJSON_GetObjectItem(k, "registers_per_thread")->valueint;
        (*kernels)[i].stream_id = cJSON_GetObjectItem(k, "stream_id")->valueint;
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
