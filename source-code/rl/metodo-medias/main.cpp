#include <iostream>
#include <armadillo>
#include <fstream>
#include <vector>

using namespace arma;
using namespace std;

// Estrutura para representar um Braço do Bandit
struct Arm {
    double mu;
    double sigma;
};
// Função para simular um agente com uma seed específica
vec run_simulation(double epsilon, int seed) {
    // Configuração conforme Figura fig:k-armed-bandit do texto
    vector<Arm> arms = {
        {0.0, 1.0},   // a1
        {-1.5, 0.5},  // a2
        {2.5, 1.5},   // a3
        {1.0, 0.8}    // a4
    };

    int k = arms.size();
    int n_steps = 1000;
    
    arma_rng::set_seed(seed);

    vec q_estimates = zeros<vec>(k); 
    vec n_counts = zeros<vec>(k);    
    vec cumulative_reward(n_steps);
    double total_sum = 0.0;

    // Inicialização Otimista ou Forçada?
    // Para ser justo com o código anterior (agente-ingenuo), vamos usar a mesma inicialização:
    // Uma vez cada braço para garantir q_estimates iniciais não nulos/enviesados por zero
    for(int i = 0; i < k; ++i) {
        double reward = randn() * arms[i].sigma + arms[i].mu;
        q_estimates[i] = reward; 
        n_counts[i] = 1;
        total_sum += reward;
        cumulative_reward[i] = total_sum;
    }

    // Loop principal
    for(int t = k; t < n_steps; ++t) {
        uword action;
        
        // Estratégia epsilon-greedy
        if(randu() < epsilon) {
            // Exploração: escolhe um braço aleatório
            action = randi(distr_param(0, k-1));
        } else {
            // Exploração: escolhe o melhor braço até agora
            action = q_estimates.index_max();
        }
        
        // Executa ação
        double reward = randn() * arms[action].sigma + arms[action].mu;

        // Atualiza Q (Sample Average)
        n_counts[action]++;
        q_estimates[action] = q_estimates[action] + 0.9 * (reward - q_estimates[action]);

        // Registra
        total_sum += reward;
        cumulative_reward[t] = total_sum;
    }
    
    return cumulative_reward;
}

// Função para executar múltiplas simulações e retornar a média
vec run_multiple_simulations(double epsilon, int num_runs) {
    int n_steps = 1000;
    mat results(n_steps, num_runs);
    
    for(int run = 0; run < num_runs; ++run) {
        vec simulation = run_simulation(epsilon, 42 + run); // Seeds diferentes
        results.col(run) = simulation;
    }
    
    // Calcula a média de todas as execuções
    vec average = mean(results, 1);
    return average;
}

int main() {
    // Parâmetros
    int num_runs = 30;  // Número de execuções com seeds diferentes

    // 1. Agente Guloso (Greedy, epsilon = 0.0) - MÉDIA
    vec reward_greedy = run_multiple_simulations(0.0, num_runs);

    // 2. Agente Epsilon-Greedy (epsilon = 0.1) - MÉDIA
    vec reward_epsilon = run_multiple_simulations(0.1, num_runs);

    // Diagnóstico
    cout << "\n=== DIAGNÓSTICO ===" << endl;
    cout << "Recompensa final (greedy): " << reward_greedy[reward_greedy.n_elem - 1] << endl;
    cout << "Recompensa final (epsilon-greedy): " << reward_epsilon[reward_epsilon.n_elem - 1] << endl;
    cout << "Diferença: " << (reward_epsilon[reward_epsilon.n_elem - 1] - reward_greedy[reward_greedy.n_elem - 1]) << endl;
    cout << "==================\n" << endl;

    // Exportar CSV combinado
    ofstream file("average_methods_results.csv");
    file << "step,greedy_reward,epsilon_greedy_reward" << endl;
    
    int n_steps = reward_greedy.n_elem;
    for(int t = 0; t < n_steps; t += 5) { // Downsampling
        file << t << "," 
             << reward_greedy[t] << "," 
             << reward_epsilon[t] << endl;
    }
    file.close();

    cout << "Simulação com " << num_runs << " execuções concluída." << endl;
    cout << "Dados (média) salvos em 'average_methods_results.csv'." << endl;

    return 0;
}
