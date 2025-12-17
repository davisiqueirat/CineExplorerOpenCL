__kernel void calculate_relevance(
    __global float* movie_ratings,   // Input: Nota do banco de dados
    __global int* movie_genres,      // Input: ID do gênero
    __global float* final_scores,    // Output: Score calculado para o usuário
    const int user_fav_genre,        // Input: Preferência do usuário (ex: Ação)
    const int num_movies             // Input: Quantidade total de filmes
) {
    // O get_global_id pega o índice único da thread. 
    // Se lançarmos 1 milhão de threads, cada uma terá um ID de 0 a 999.999.
    int i = get_global_id(0);

    // Garante que a thread não tente ler memória que não existe
    if (i >= num_movies) {
        return;
    }

    // 3. Leitura de Dados (Da memória global da GPU)
    float score = movie_ratings[i];
    int genre = movie_genres[i];
    
    // Se for o gênero favorito, aumenta muito a relevância
    if (genre == user_fav_genre) {
        score += 5.0f; // Bônus de relevância
    } 
    // Se for um gênero "próximo" (ex: ID + 1), dá um bônus menor (exemplo de lógica fuzzy)
    else if (genre == user_fav_genre + 1 || genre == user_fav_genre - 1) {
        score += 1.5f;
    }

    // Aplica um multiplicador final 
    score = score * 1.1f;

    // 5. Escrita do Resultado
    final_scores[i] = score;
}