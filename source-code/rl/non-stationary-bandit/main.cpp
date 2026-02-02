#include <armadillo>
#include <fstream>
#include <iostream>
#include <vector>

using namespace arma;
using namespace std;

// Estrutura para representar um Braço do Bandit NÃO-ESTACIONÁRIO
struct NonStationaryArm {
  double mu;
  double sigma;

  // Aplica um random walk na média
  void drift(double step_size = 0.01) { mu += randn() * step_size; }

  // Aplica um shift grande (mudança de regime)
  void regime_shift(double magnitude = 1.0) { mu += randn() * magnitude; }
};

// Função para simular um agente com uma seed específica
// strategy: 0=Greedy, 1=Epsilon-Greedy (sample average), 2=Constant-Alpha
mat run_simulation(int strategy, double param, int seed) {
  // Configuração inicial conforme Figura fig:k-armed-bandit do texto
  vector<NonStationaryArm> arms = {
      {0.0, 1.0},  // a1
      {-1.5, 0.5}, // a2
      {2.5, 1.5},  // a3
      {1.0, 0.8}   // a4
  };

  int k = arms.size();
  int n_steps = 5000; // Aumentado para ver o efeito de longo prazo

  arma_rng::set_seed(seed);

  vec q_estimates = zeros<vec>(k);
  vec n_counts = zeros<vec>(k);
  vec cumulative_reward(n_steps);
  vec instantaneous_reward(n_steps); // Recompensa por step (não acumulada)
  double total_sum = 0.0;

  // Inicialização forçada (uma vez cada braço)
  for (int i = 0; i < k; ++i) {
    double reward = randn() * arms[i].sigma + arms[i].mu;
    q_estimates[i] = reward;
    n_counts[i] = 1;
    total_sum += reward;
    cumulative_reward[i] = total_sum;
    instantaneous_reward[i] = reward;
  }

  // Loop principal
  for (int t = k; t < n_steps; ++t) {
    // === NON-STATIONARITY: Random Walk CONTÍNUO ===
    // A CADA step, aplica drift nos braços (muito mais não-estacionário!)
    for (auto &arm : arms) {
      arm.drift(0.1); // Step size aumentado de 0.05 para 0.1
    }

    // === REGIME SHIFTS: Mudanças bruscas a cada 1000 steps ===
    // Isso força mudanças claras na ação ótima
    if (t % 1000 == 0 && t > 0) {
      for (auto &arm : arms) {
        arm.regime_shift(2.0); // Shift brusco de magnitude 2.0
      }
    }

    uword action;

    // Estratégia de seleção de ação
    if (strategy == 0) {
      // Greedy puro (epsilon = 0)
      action = q_estimates.index_max();
    } else if (strategy == 1) {
      // Epsilon-Greedy com sample average
      double epsilon = param;
      if (randu() < epsilon) {
        action = randi(distr_param(0, k - 1));
      } else {
        action = q_estimates.index_max();
      }
    } else {
      // Epsilon-Greedy com constant alpha
      double epsilon = 0.1; // Fixo para comparação justa
      if (randu() < epsilon) {
        action = randi(distr_param(0, k - 1));
      } else {
        action = q_estimates.index_max();
      }
    }

    // Executa ação
    double reward = randn() * arms[action].sigma + arms[action].mu;

    // Atualiza Q
    n_counts[action]++;

    if (strategy == 2) {
      // Constant alpha (exponential moving average)
      double alpha = param;
      q_estimates[action] =
          q_estimates[action] + alpha * (reward - q_estimates[action]);
    } else {
      // Sample average (incremental)
      q_estimates[action] =
          q_estimates[action] +
          (1.0 / n_counts[action]) * (reward - q_estimates[action]);
    }

    // Registra
    total_sum += reward;
    cumulative_reward[t] = total_sum;
    instantaneous_reward[t] = reward;
  }

  // Retorna matriz com cumulative_reward e instantaneous_reward
  mat result(n_steps, 2);
  result.col(0) = cumulative_reward;
  result.col(1) = instantaneous_reward;
  return result;
}

// Função para executar múltiplas simulações e retornar a média
mat run_multiple_simulations(int strategy, double param, int num_runs) {
  int n_steps = 5000;
  mat cumulative_results(n_steps, num_runs);
  mat instantaneous_results(n_steps, num_runs);

  for (int run = 0; run < num_runs; ++run) {
    mat simulation = run_simulation(strategy, param, 42 + run);
    cumulative_results.col(run) = simulation.col(0);
    instantaneous_results.col(run) = simulation.col(1);
  }

  // Calcula a média de todas as execuções
  vec average_cumulative = mean(cumulative_results, 1);
  vec average_instantaneous = mean(instantaneous_results, 1);

  mat result(n_steps, 2);
  result.col(0) = average_cumulative;
  result.col(1) = average_instantaneous;
  return result;
}

int main() {
  // Parâmetros
  int num_runs = 50; // Número de execuções com seeds diferentes

  cout << "Iniciando simulações em ambiente FORTEMENTE NÃO-ESTACIONÁRIO..."
       << endl;
  cout << "- Random walk contínuo (σ=0.1 por step)" << endl;
  cout << "- Regime shifts a cada 1000 steps (magnitude 2.0)" << endl;
  cout << "- 5000 steps totais\n" << endl;

  // 1. Agente Greedy (epsilon = 0.0)
  cout << "Executando Greedy..." << endl;
  mat result_greedy = run_multiple_simulations(0, 0.0, num_runs);

  // 2. Agente Epsilon-Greedy com Sample Average (epsilon = 0.1)
  cout << "Executando Epsilon-Greedy (sample average)..." << endl;
  mat result_epsilon_sample = run_multiple_simulations(1, 0.1, num_runs);

  // 3. Agente Epsilon-Greedy com Constant Alpha (alpha = 0.1)
  cout << "Executando Epsilon-Greedy (constant alpha)..." << endl;
  mat result_constant_alpha = run_multiple_simulations(2, 0.1, num_runs);

  // Diagnóstico
  cout << "\n=== DIAGNÓSTICO ===" << endl;
  cout << "Recompensa final (greedy): "
       << result_greedy(result_greedy.n_rows - 1, 0) << endl;
  cout << "Recompensa final (epsilon-sample): "
       << result_epsilon_sample(result_epsilon_sample.n_rows - 1, 0) << endl;
  cout << "Recompensa final (constant-alpha): "
       << result_constant_alpha(result_constant_alpha.n_rows - 1, 0) << endl;

  // Recompensa média nos últimos 1000 steps (mais importante!)
  int last_n = 100; // Últimos 1000 steps (downsampling de 10)
  cout << "\nRecompensa MÉDIA (últimos 1000 steps - MAIS IMPORTANTE!):" << endl;
  double greedy_avg = mean(result_greedy(
      span(result_greedy.n_rows - last_n, result_greedy.n_rows - 1), 1));
  double epsilon_avg =
      mean(result_epsilon_sample(span(result_epsilon_sample.n_rows - last_n,
                                      result_epsilon_sample.n_rows - 1),
                                 1));
  double alpha_avg =
      mean(result_constant_alpha(span(result_constant_alpha.n_rows - last_n,
                                      result_constant_alpha.n_rows - 1),
                                 1));

  cout << "  Greedy:              " << greedy_avg << endl;
  cout << "  ε-Greedy (sample):   " << epsilon_avg << endl;
  cout << "  ε-Greedy (α const):  " << alpha_avg << endl;
  cout << "==================\n" << endl;

  // Exportar CSV combinado
  ofstream file("non_stationary_comparison.csv");
  file << "step,greedy_reward,epsilon_sample_reward,constant_alpha_reward,"
          "greedy_inst,epsilon_inst,alpha_inst"
       << endl;

  int n_steps = result_greedy.n_rows;
  for (int t = 0; t < n_steps; t += 10) { // Downsampling
    file << t << "," << result_greedy(t, 0) << ","
         << result_epsilon_sample(t, 0) << "," << result_constant_alpha(t, 0)
         << "," << result_greedy(t, 1) << "," << result_epsilon_sample(t, 1)
         << "," << result_constant_alpha(t, 1) << endl;
  }
  file.close();

  cout << "Simulação com " << num_runs << " execuções concluída." << endl;
  cout << "Dados salvos em 'non_stationary_comparison.csv'." << endl;

  return 0;
}
