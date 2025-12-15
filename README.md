## Sistema de Recomendação de Filmes com OpenCL (High Performance) 

> Projeto Acadêmico: Para cadeira de Fábrica de Software. Usando Computação Paralela
> Integração: Backend Simulado para Projeto React/Firebase 

Este repositório contém a implementação em C e OpenCL de um motor de recomendação de filmes. O objetivo é demonstrar como o processamento massivo em GPU (Parallel Computing) resolve gargalos de performance que seriam impossíveis de tratar apenas com JavaScript no Frontend ou com processamento serial tradicional. 

---

##  O Que Está Sendo Simulado? 

Como este é um módulo de teste de performance (Benchmark), isolamos a lógica do banco de dados real (Firebase) para focar puramente na capacidade de cálculo.

### 1. Mock de Dados (O "Banco de Dados" Virtual)
O código gera arrays em memória RAM simulando um snapshot do banco de dados de produção:
- Volume: 10 Milhões de filmes (Simulação de Big Data). 
- Dados: 
  - `movie_ratings`: Notas originais (simulando a média do TMDB/IMDb).
  - `movie_genres`: IDs numéricos representando gêneros (ex: 28 = Ação).

 2. A Lógica de Recomendação (O Algoritmo) 
O kernel na GPU executa a seguinte lógica matemática para cada um dos 10 milhões de filmes simultaneamente: 

1.  Leitura Base: O algoritmo lê a nota original do filme (ex: 8.5). 
2.  Verificação de Preferência (Filtro Personalizado):** 
    * Se o gênero do filme for **EXATAMENTE** o favorito do usuário:
        * Ação:*Soma +5.0 pontos ao score. 
    * Se o gênero for APROXIMADO (Lógica Fuzzy - IDs vizinhos): 
        * Ação:** Soma +1.5 pontos** ao score. 
3.  Ponderação Final: 
    * O resultado é multiplicado por **1.1x** (peso do algoritmo) para gerar o `final_score`.

> Exemplo Prático: 
> Um filme com nota 6.0 (mediano), mas que é do gênero favorito do usuário, passará a ter nota **(6.0 + 5.0) *1.1 = 12.1. Ele passará na frente de um filme nota 9.0 que não é do gênero favorito. 

---

##  Benchmark e Resultados 

Os testes comparam a execução da **mesma lógica matemática** rodando sequencialmente na CPU versus rodando paralelamente na GPU.

**Ambiente de Teste:**
- **CPU:** Intel Core i5-9400F (2.90GHz) - *Processamento Serial*
- **GPU:** NVIDIA GeForce GTX 1660 Super (1408 CUDA Cores) - *Processamento Paralelo*

### Log de Execução 


--- BENCHMARK: CPU vs GPU (OpenCL) ---
Processando 10000000 filmes...

1. Iniciando processamento na CPU (Serial)...
   -> CPU Concluido em: 0.4500 segundos

2. Iniciando processamento na GPU (Paralelo)...
   -> GPU Concluido em: 0.0400 segundos

======================================================
>>> SPEEDUP (Aceleração): 11.25x mais rapido na GPU <<<
======================================================
