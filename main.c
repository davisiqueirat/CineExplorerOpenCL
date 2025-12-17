#include <stdio.h>
#include <stdlib.h>
#include <time.h>     // Para medir o tempo e gerar aleatórios
#include <math.h>     // Para funcoes matematicas 
#include <CL/cl.h>    // Biblioteca OpenCL

#define MAX_SOURCE_SIZE (0x100000)

// -------------------------------------------------------------------------
// FUNÇÃO SERIAL (RODA NA CPU) - Para comparação
// -------------------------------------------------------------------------
void calculate_serial_host(float* ratings, int* genres, float* scores, int user_fav, int n) {
    for(int i = 0; i < n; i++) {
        float score = ratings[i];
        int genre = genres[i];

        // Mesma lógica do Kernel OpenCL
        if (genre == user_fav) {
            score += 5.0f;
        } 
        else if (genre == user_fav + 1 || genre == user_fav - 1) {
            score += 1.5f;
        }

        score = score * 1.1f;
        scores[i] = score;
    }
}

int main() {
    // -------------------------------------------------------------------------
    // CONFIGURAÇÃO DOS DADOS
    // -------------------------------------------------------------------------
    // 10 MILHÕES para a GPU mostrar poder real. 
    
    int num_movies = 10000000; 
    int user_fav_genre = 28;

    size_t datasize_ratings = sizeof(float) * num_movies;
    size_t datasize_genres  = sizeof(int) * num_movies;
    size_t datasize_scores  = sizeof(float) * num_movies;

    // Alocação de Memória
    float *movie_ratings = (float*)malloc(datasize_ratings);
    int   *movie_genres  = (int*)malloc(datasize_genres);
    float *gpu_results   = (float*)malloc(datasize_scores); // Resultado da GPU
    float *cpu_results   = (float*)malloc(datasize_scores); // Resultado da CPU

    // Preenchendo dados aleatórios
    srand(time(NULL)); 
    printf("--- BENCHMARK: CPU vs GPU (OpenCL) ---\n");
    printf("Processando %d filmes...\n\n", num_movies);

    for(int i = 0; i < num_movies; i++) {
        movie_ratings[i] = ((float)(rand() % 90) / 10.0f) + 1.0f;
        movie_genres[i] = (rand() % 15) + 20; 
    }

    // -------------------------------------------------------------------------
    // MEDINDO O TEMPO DA CPU (SERIAL)
    // -------------------------------------------------------------------------
    printf("1. Iniciando processamento na CPU (Serial)...\n");
    clock_t start_cpu = clock();
    
    calculate_serial_host(movie_ratings, movie_genres, cpu_results, user_fav_genre, num_movies);
    
    clock_t end_cpu = clock();
    double time_cpu = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("   -> CPU Concluido em: %.4f segundos\n\n", time_cpu);

    // -------------------------------------------------------------------------
    // PREPARANDO E MEDINDO O TEMPO DA GPU (OPENCL)
    // -------------------------------------------------------------------------
    printf("2. Iniciando processamento na GPU (Paralelo)...\n");

    // Boilerplate OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices, ret_num_platforms;
    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Carrega Kernel
    FILE *fp = fopen("recommender.cl", "r");
    if (!fp) { fprintf(stderr, "Erro ao abrir recommender.cl\n"); return 1; }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculate_relevance", NULL);

    // Buffers
    cl_mem mem_ratings = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize_ratings, NULL, NULL);
    cl_mem mem_genres  = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize_genres, NULL, NULL);
    cl_mem mem_scores  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize_scores, NULL, NULL);

    // Copia dados para GPU
    clEnqueueWriteBuffer(command_queue, mem_ratings, CL_TRUE, 0, datasize_ratings, movie_ratings, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, mem_genres, CL_TRUE, 0, datasize_genres, movie_genres, 0, NULL, NULL);

    // Argumentos
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_ratings);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_genres);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_scores);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&user_fav_genre);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&num_movies);

    size_t global_item_size = num_movies;
    size_t local_item_size = 1; // Deixe NULL se quiser automatico, 64 se quiser otimizar

    // --- CRONÔMETRO GPU (Apenas o processamento) ---
    
    clock_t start_gpu = clock();

    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clFinish(command_queue); // Espera a GPU terminar de verdade

    clock_t end_gpu = clock();
    // ---------------------------------------------

    // Traz o resultado de volta
    clEnqueueReadBuffer(command_queue, mem_scores, CL_TRUE, 0, datasize_scores, gpu_results, 0, NULL, NULL);

    double time_gpu = ((double) (end_gpu - start_gpu)) / CLOCKS_PER_SEC;
    printf("   -> GPU Concluido em: %.4f segundos\n", time_gpu);

    // -------------------------------------------------------------------------
    //  RESULTADOS FINAIS
    // -------------------------------------------------------------------------
    printf("\n======================================================\n");
    printf("RESULTADOS DO COMPARATIVO\n");
    printf("======================================================\n");
    printf("Tempo Serial (CPU):   %.4f s\n", time_cpu);
    printf("Tempo Paralelo (GPU): %.4f s\n", time_gpu);
    
    double speedup = time_cpu / time_gpu;
    printf("\n>>> SPEEDUP (Aceleração): %.2fx mais rapido na GPU <<<\n", speedup);
    printf("======================================================\n");

    // Validação (Verifica se GPU calculou igual a CPU)
    int erros = 0;
    for(int i = 0; i < num_movies; i++) {
        // Compara com uma margem de erro pequena (float precision)
        if(fabs(cpu_results[i] - gpu_results[i]) > 0.001) {
            erros++;
            if(erros < 5) printf("Erro no indice %d: CPU=%.2f GPU=%.2f\n", i, cpu_results[i], gpu_results[i]);
        }
    }

    
    if(erros == 0) printf("\nVerificacao de Corretude: SUCESSO! Resultados identicos.\n");
    else printf("\nVerificacao de Corretude: FALHA! Encontrados %d erros.\n", erros);


    
    // Limpeza
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(mem_ratings);
    clReleaseMemObject(mem_genres);
    clReleaseMemObject(mem_scores);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(source_str);
    free(movie_ratings);
    free(movie_genres);
    free(gpu_results);
    free(cpu_results);

    return 0;
}

    //RESULTADO DA EXECUÇÃO (Ambiente de Teste)
    //Hardware: Intel Core i5-9400F + NVIDIA GTX 1660 Super
    // OS: Windows 10 (CUDA Toolkit 12.0)

  //     BENCHMARK: CPU vs GPU (OpenCL) 
      //    Processando 10000000 filmes...

      //    1. Iniciando processamento na CPU (Serial)
      //    -> CPU Concluido em: 0.4500 segundos

      //     2. Iniciando processamento na GPU (Paralelo)
      //     -> GPU Concluido em: 0.0400 segundos

      // RESULTADOS DO COMPARATIVO
    // Tempo Serial (CPU):   0.4500 s
    // Tempo Paralelo (GPU): 0.0400 s

      //      >>> SPEEDUP (Aceleração): 11.25x mais rapido na GPU <<<
      //     Verificacao de Corretude: SUCESSO! Resultados identicos.
     