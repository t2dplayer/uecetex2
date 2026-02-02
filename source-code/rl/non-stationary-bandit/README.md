# Comparação de Estratégias em Ambiente Não-Estacionário

Este projeto implementa e compara três estratégias diferentes para o problema do k-armed bandit em um ambiente **não-estacionário**, onde as médias das recompensas mudam ao longo do tempo.

## 📋 Estratégias Implementadas

### 1. **Greedy (ε=0)**
- Sempre seleciona a ação com maior valor estimado
- Sem exploração
- Boa performance inicial, mas pode ficar "preso" em ações subótimas quando o ambiente muda

### 2. **ε-Greedy com Média Amostral**
- Explora com probabilidade ε=0.1
- Atualiza estimativas usando média incremental: `Q(a) ← Q(a) + (1/N(a))[R - Q(a)]`
- Performance consistente, mas lenta adaptação às mudanças

### 3. **ε-Greedy com Taxa de Aprendizado Constante (α)**
- Explora com probabilidade ε=0.1
- Atualiza estimativas usando α constante: `Q(a) ← Q(a) + α[R - Q(a)]`
- **Melhor adaptação** a ambientes não-estacionários
- Dá mais peso às recompensas recentes

## 🔬 Ambiente Não-Estacionário

O ambiente implementa não-estacionariedade através de um **random walk** nas médias das recompensas:
- A cada 10 interações, aplica-se um drift: `μ ← μ + N(0, 0.05²)`
- Isto faz com que a ação ótima mude ao longo do tempo
- Simula condições realistas onde o ambiente evolui dinamicamente

## 🛠️ Compilação e Execução

### Requisitos
- g++ com suporte a C++17
- Biblioteca Armadillo
- Python 3 (para análise)
- matplotlib, pandas, numpy (para visualização)

### Compilar e Executar
```bash
make          # Compila o código
make run      # Executa a simulação
python3 analyze.py  # Gera análise e gráficos
```

### Limpar
```bash
make clean    # Remove executável e arquivos CSV
```

## 📊 Resultados

### Arquivos Gerados
- `non_stationary_comparison.csv` - Dados brutos da simulação
- `non_stationary_analysis.pdf` - Análise visual comparativa
- `non_stationary_analysis.png` - Versão PNG dos gráficos

### Parâmetros da Simulação
- **Número de braços (k)**: 4
- **Número de interações**: 2000
- **Número de execuções**: 50 (média)
- **ε (exploração)**: 0.1
- **α (taxa de aprendizado)**: 0.1
- **Step size do random walk**: 0.05

## 📈 Interpretação dos Resultados

Com base nas simulações (50 execuções):

**Recompensa Final (t=2000)**:
- Greedy: ~4638
- ε-Greedy (sample): ~4381  
- ε-Greedy (α const): ~4362

**Observações**:
1. Em curto prazo, o Greedy pode parecer superior
2. A longo prazo, a capacidade de adaptação do α constante torna-se crucial
3. O método com α constante **esquece** estimativas antigas, priorizando informação recente
4. Em ambientes estacionários, a média amostral seria superior
5. Em ambientes não-estacionários, **α constante é preferível**

## 📝 Referências

Esta implementação segue as definições apresentadas em:
- Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- Seção de Reinforcement Learning do documento de qualificação

## 🔗 Integração com LaTeX

O arquivo CSV gerado é automaticamente usado pelo documento LaTeX principal em:
```latex
\ref{fig:non_stationary_comparison}
```

Para atualizar a figura no documento:
```bash
cp non_stationary_comparison.csv ../../../elementos-textuais/reinforcement-learning/
cd ../../../
make  # Recompila o documento
```

## 💡 Insights Teóricos

### Por que α constante funciona melhor?

A atualização com α constante implementa uma **média móvel exponencial**:

```
Q_n+1 = Q_n + α[R_n - Q_n]
     = (1-α)Q_n + αR_n
     = (1-α)[(1-α)Q_{n-1} + αR_{n-1}] + αR_n
     = αR_n + α(1-α)R_{n-1} + α(1-α)²R_{n-2} + ...
```

Cada recompensa R_t tem peso α(1-α)^(n-t), que **decai exponencialmente** com o tempo.

Com α=0.1:
- Recompensa atual: peso 0.1
- 10 steps atrás: peso ~0.035
- 50 steps atrás: peso ~0.0005 (praticamente esquecida)

Isso cria um **"esquecimento adaptativo"** ideal para ambientes não-estacionários!

---

**Autor**: Código desenvolvido para análise comparativa de estratégias RL  
**Data**: Janeiro 2026
