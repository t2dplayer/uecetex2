#include <iostream>
#include <armadillo>
#include <fstream>

using namespace arma;
using namespace std;

// Estrutura para representar um Braço do Bandit
struct Arm {
    double mu;
    double sigma;
};

int main() {
    // Configuração conforme Figura fig:k-armed-bandit do texto
    // a1: mu=0, sigma=1
    // a2: mu=-1.5, sigma=0.5
    // a3: mu=2.5, sigma=1.5 (Melhor média, alta variância)
    // a4: mu=1.0, sigma=0.8 (Média boa, baixa variância)
    vector<Arm> arms = {
        {0.0, 1.0},   // a1
        {-1.5, 0.5},  // a2
        {2.5, 1.5},   // a3
        {1.0, 0.8}    // a4
    };

    int k = 4;
    int n_steps = 1000; // Horizonte de tempo

    // Fixar semente para garantir que o cenário descrito no texto ocorra
    // (onde uma ação subótima pode parecer melhor inicialmente devido à variância)
    arma_rng::set_seed(1078);

    vec q_estimates = zeros<vec>(k); // Estimativa de valor Q(a)
    vec n_counts = zeros<vec>(k);    // Contagem de vezes que a ação foi escolhida
    vec cumulative_reward(n_steps);
    vec actions(n_steps);
    double total_sum = 0.0;

    // 1. Fase Inicial: Escolhe uma de cada vez (Exploração forçada mínima)
    for(int i = 0; i < k; ++i) {
        double reward = randn() * arms[i].sigma + arms[i].mu;

        q_estimates[i] = reward; // Primeira estimativa é o próprio valor observado
        n_counts[i] = 1;

        total_sum += reward;
        cumulative_reward[i] = total_sum;
    }

    // 2. Fase Greedy: Apenas Explotação (escolhe o maior Q estimado atual)
    for(int t = k; t < n_steps; ++t) {
        // Encontra ação com maior valor estimado (Argmax)
        uword action = q_estimates.index_max();

        // Executa ação no ambiente
        double reward = randn() * arms[action].sigma + arms[action].mu;

        // Atualiza estimativa de valor (Média Incremental)
        n_counts[action]++;
        q_estimates[action] = q_estimates[action] + (1.0 / n_counts[action]) * (reward - q_estimates[action]);

        // Registra
        total_sum += reward;
        cumulative_reward[t] = total_sum;
        actions[t] = action;
    }

    // 3. Exportar para CSV
    ofstream file("bandit_results.csv");
    file << "step,accumulated_reward,action" << endl;
    for(int t = 0; t < n_steps; t += 5) { // Downsampling para gráfico mais leve
        file << t << ","
            << cumulative_reward[t] << ","
            << actions[t] << endl;
    }
    file.close();

    cout << "Simulação concluída. Dados salvos em 'bandit_results.csv'." << endl;
    return 0;
}
