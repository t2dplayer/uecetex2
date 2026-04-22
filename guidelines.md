# Diretrizes de Escrita

Corpus: `t2dplayer/uecetex2`
Projeto: `/home/sergiosvieira/Documentos/2025.2/qualifica/uecetex2`

## Regras de Estilo LaTeX

### Siglas e Abreviaturas

- **Primeira Ocorrência:** Sempre usar o formato "Texto Completo (SIGLA)". 
  - ✅ Correto: "O Aprendizado por Reforço Profundo (DRL) ganhou tração substancial..."
  - ❌ Errado: "O DRL ganhou tração substancial..." (se nunca citado antes)
  - ❌ Errado: "O Aprendizado por Reforço Profundo ganhou tração substancial..." (sem inserir a sigla se ela for ser usada)
- **Ocorrências Subsequentes:** Se já apresentou o texto completo com a sigla anteriormente no texto, utilizar **apenas a sigla** nas próximas vezes. NÃO repita o texto completo (muito menos na forma "Texto Completo (SIGLA)" novamente) em capítulos que já definiram a sigla.
  - ✅ Correto: "O DRL mitiga esse desafio..."
  - ❌ Errado: "O Aprendizado por Reforço Profundo mitiga esse desafio..." (já citado e definido anteriormente).

### Pontuação e Estrutura de Frases

- **Quebras de Linha (`\\n`):** Em arquivos LaTeX, faça quebra de linha **apenas quando for criar um novo parágrafo** (ou seja, duas quebras consecutivas `\\n\\n` para iniciar um parágrafo novo). Não utilize a técnica de quebrar linhas a cada vírgula (*semantic linefeeds*) ou a cada 80 colunas no texto corrido. As frases de um mesmo parágrafo devem permanecer contínuas na mesma linha lógica do arquivo.

- **Travessão (`---`) NÃO deve ser usado.** Para explicar ou intercalar informações, usar **vírgula**. Parênteses `()` para informações muito curtas. Ponto e vírgula (`;`) para separar itens dentro de frases.
  - ❌ Errado: "o estado TOPSIS carrega informação útil—propriedade crítica quando as estimativas são ruidosas"
  - ✅ Correto: "o estado TOPSIS carrega informação útil, propriedade crítica quando as estimativas são ruidosas"

- **`\emph{}`, `\textit{}` e `\textbf{}` NÃO devem ser usados em parágrafos de texto corrido.** Esses comandos só são permitidos em:
  - Legendas de figuras e tabelas;
  - Títulos de itens de listas (`\item`);
  - Equações e definições matemáticas formais.
  - ❌ Errado: "o ganho é \emph{monótono}: o TOPSIS nunca prejudica..."
  - ✅ Correto: "o ganho é monótono, pois o TOPSIS nunca prejudica..."

- **Dois-pontos (`:`) NÃO devem ser usados para explicar.** O uso é permitido apenas para **exemplificar**, ou seja, introduzir uma lista de exemplos concretos após um termo genérico.
  - ✅ Correto: "Os critérios avaliados são: latência, energia e distância."
  - ❌ Errado: "O segundo eixo representa uma mudança de paradigma: as limitações do DSRC impulsionaram..."
  - Quando a intenção for explicar ou desenvolver um raciocínio, usar conectivos como "pois", "porque", "já que", "visto que", ou reestruturar em nova oração.

### Critérios do TOPSIS (modelo SAC-TOPSIS)

Os **3 critérios reais** do módulo TOPSIS implementado são:
1. Tempo de conclusão estimado (JCT)
2. Consumo energético
3. Distância ao nó de computação

### Terminologia de Candidatos ao Descarregamento

Usar "servidor de borda (VEC) e veículos vizinhos" em vez de "RSU e veículos vizinhos", pois o ambiente simulado usa um Edge Server (Dispositivo de Borda) — não uma RSU stricto sensu.

### Arquitetura por Etapa

- **Etapa intermediária (esta qualificação):** SAC-TOPSIS
- **Trabalho futuro (tese final):** SAC-Fuzzy-TOPSIS + validação NS-3 3.42

Não descrever o capítulo de Proposta como "SAC-Fuzzy-TOPSIS". O capítulo apresenta SAC-TOPSIS.

### Contagens Verificadas

- **Heurísticas avaliadas:** 7 (VEC, Random, Round-Robin, Gini, Jain, Greedy-Fair, Greedy-Unfair)
- **Famílias de DRL:** 5 (SAC, PPO, TD3, DQN, A2C)

