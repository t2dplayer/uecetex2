# Respostas para Questões sobre RL

## Questão 1: RL Offline

**Pergunta**: Você considera explicitamente RL offline em algum ponto do trabalho? Se sim, ajuste a frase sobre dataset fixo para evitar generalização.

**Resposta**: 
Não, o trabalho **não** considera RL offline. O foco está em RL clássico (online), onde o agente aprende através de interação direta com o ambiente. 

**Ação Corretiva**: 
A frase original "não há um dataset fixo entregue ao algoritmo" foi ajustada para "No aprendizado por reforço **clássico**, a criação do modelo é realizada de forma diferente..." para deixar explícito que existe RL offline (que usa datasets fixos, como D4RL, Atari datasets, etc.), mas não é o paradigma abordado neste trabalho.

**Texto Atualizado** (linha ~151):
```latex
No aprendizado por reforço clássico, a criação do modelo é realizada de forma 
diferente, pois não há um dataset fixo entregue ao algoritmo. O aprendizado 
emerge da experiência direta de um agente (algoritmo de decisão) com um 
ambiente dinâmico (o mundo e suas regras).
```

---

## Questão 2: Metodologia de Simulação

**Pergunta**: Nos gráficos de comparação (Greedy vs ε-Greedy etc.), os resultados são de uma execução ou média de múltiplas execuções? Quantas? Qual seed?

**Resposta**:
- **Número de execuções**: 50 execuções independentes
- **Seeds**: Base seed = 42, incrementado para cada execução (42, 43, 44, ..., 91)
- **Agregação**: Média aritmética das 50 execuções
- **Implementação**: Ver `run_multiple_simulations()` em `/source-code/rl/non-stationary-bandit/main.cpp`

**Código Relevante** (main.cpp, linhas 130-148):
```cpp
mat run_multiple_simulations(int strategy, double param, int num_runs) {
  int n_steps = 5000;
  mat cumulative_results(n_steps, num_runs);
  mat instantaneous_results(n_steps, num_runs);

  for (int run = 0; run < num_runs; ++run) {
    mat simulation = run_simulation(strategy, param, 42 + run); // Seeds: 42-91
    cumulative_results.col(run) = simulation.col(0);
    instantaneous_results.col(run) = simulation.col(1);
  }

  // Calcula a média de todas as execuções
  vec average_cumulative = mean(cumulative_results, 1);
  vec average_instantaneous = mean(instantaneous_results, 1);
  
  // ...
}
```

**Ação Corretiva**:
Adicionar nota de rodapé ou texto explicativo na caption da figura mencionando a metodologia de simulação.

---

## Questão 3: Especificação do Ambiente Não-Estacionário

**Pergunta**: No cenário não estacionário: o "random walk" e "regime shifts" são implementados exatamente como? (magnitude do passo, frequência dos shifts). Uma linha de especificação ajuda.

**Resposta Detalhada**:

### Random Walk Contínuo
- **Frequência**: A cada step (interação)
- **Magnitude**: σ = 0.1 (desvio padrão)
- **Implementação**: `μ ← μ + N(0, 0.1²)` para cada braço
- **Código**: Linha 62-65 em main.cpp

```cpp
// A CADA step, aplica drift nos braços
for (auto &arm : arms) {
  arm.drift(0.1);  // Step size = 0.1
}
```

### Regime Shifts
- **Frequência**: A cada 1000 interações (t = 1000, 2000, 3000, 4000)
- **Magnitude**: σ = 2.0 (desvio padrão)
- **Implementação**: `μ ← μ + N(0, 2.0²)` para cada braço
- **Código**: Linhas 67-72 em main.cpp

```cpp
// Mudanças bruscas a cada 1000 steps
if (t % 1000 == 0 && t > 0) {
  for (auto &arm : arms) {
    arm.regime_shift(2.0);  // Magnitude = 2.0
  }
}
```

### Parâmetros Completos do Ambiente

| Parâmetro | Valor |
|-----------|-------|
| Número de braços (k) | 4 |
| Horizonte temporal | 5000 steps |
| Random walk σ | 0.1 por step |
| Regime shift σ | 2.0 a cada 1000 steps |
| Execuções (média) | 50 |
| Seeds | 42-91 |

**Ação Corretiva**:
A caption da figura já foi atualizada para incluir essas especificações:

```latex
\caption{Comparação de estratégias em ambiente fortemente não-estacionário: 
Greedy vs. ε-Greedy (média amostral) vs. ε-Greedy (taxa constante α=0.1). 
O ambiente aplica random walk contínuo (σ=0.1 por step) nas médias das 
recompensas, além de regime shifts (magnitude 2.0) a cada 1000 interações, 
tornando a ação ótima altamente variável ao longo de 5000 interações totais.}
```

---

## Melhorias Adicionais Sugeridas

### 1. Adicionar Nota de Rodapé sobre Replicabilidade

Sugestão de texto após a Figura \ref{fig:non_stationary_comparison}:

```latex
\footnote{Os resultados apresentados correspondem à média de 50 execuções 
independentes (seeds 42-91) para garantir robustez estatística. O código-fonte 
da simulação está disponível em \texttt{source-code/rl/non-stationary-bandit/}.}
```

### 2. Adicionar Tabela de Parâmetros

Opcionalmente, criar uma tabela detalhando todos os parâmetros experimentais:

```latex
\begin{table}[htbp]
    \centering
    \caption{Parâmetros da simulação não-estacionária}
    \label{tab:non_stationary_params}
    \begin{tabular}{ll}
        \toprule
        Parâmetro & Valor \\
        \midrule
        Número de braços ($k$) & 4 \\
        Horizonte temporal & 5000 steps \\
        Random walk (σ) & 0.1 por step \\
        Regime shifts (σ) & 2.0 a cada 1000 steps \\
        Taxa de exploração ($\varepsilon$) & 0.1 \\
        Taxa de aprendizado ($\alpha$) & 0.1 \\
        Execuções (média) & 50 (seeds 42-91) \\
        \bottomrule
    \end{tabular}
\end{table}
```

---

## Verificação de Reprodutibilidade

Para verificar os resultados, execute:

```bash
cd source-code/rl/non-stationary-bandit/
make clean && make && make run
```

**Resultado Esperado** (média de 50 execuções):
```
Recompensa final (greedy):              ~20,524
Recompensa final (epsilon-sample):      ~20,633
Recompensa final (constant-alpha):      ~24,898

Recompensa MÉDIA (últimos 1000 steps):
  Greedy:              ~5.67
  ε-Greedy (sample):   ~6.20
  ε-Greedy (α const):  ~7.76
```
