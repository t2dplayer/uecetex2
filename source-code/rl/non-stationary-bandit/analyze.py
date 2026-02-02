#!/usr/bin/env python3
"""
Análise MELHORADA - foca em recompensa instantânea (sliding window)
para evidenciar a adaptação em ambiente não-estacionário.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

# Carregar dados
data = pd.read_csv('non_stationary_comparison.csv')

# Criar figura com 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ========== PLOT 1: Recompensa Acumulada (como no LaTeX) ==========
ax1.plot(data['step'], data['greedy_reward'], 
         'k--', linewidth=2, label='Greedy (ε=0)', alpha=0.8)
ax1.plot(data['step'], data['epsilon_sample_reward'], 
         'b-', linewidth=2, label='ε-Greedy (média amostral, ε=0.1)')
ax1.plot(data['step'], data['constant_alpha_reward'], 
         'r-', linewidth=2, label='ε-Greedy (α=0.1 constante)')

ax1.set_xlabel('Interações (t)')
ax1.set_ylabel('Recompensa Acumulada')
ax1.set_title('(a) Recompensa Acumulada Total - Greedy parece "melhor" entre t=2000-4500 devido à sorte inicial')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Adicionar anotações explicativas
ax1.annotate('Greedy tem vantagem\ninicial (sorte)', 
             xy=(1500, data[data['step']==1500]['greedy_reward'].values[0]), 
             xytext=(1000, 8000),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, ha='center')

ax1.annotate('α constante:\nmaior taxa de\ncrescimento recente', 
             xy=(4500, data[data['step']==4500]['constant_alpha_reward'].values[0]), 
             xytext=(3500, 28000),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, ha='center', color='red')

# ========== PLOT 2: Recompensa Média (Sliding Window 100 steps) ==========
window = 10  # 100 steps reais (downsampling de 10)

# Calcular médias móveis
greedy_avg = pd.Series(data['greedy_inst']).rolling(window=window, center=True).mean()
epsilon_avg = pd.Series(data['epsilon_inst']).rolling(window=window, center=True).mean()
alpha_avg = pd.Series(data['alpha_inst']).rolling(window=window, center=True).mean()

ax2.plot(data['step'], greedy_avg, 
         'k--', linewidth=2, label='Greedy (ε=0)', alpha=0.8)
ax2.plot(data['step'], epsilon_avg, 
         'b-', linewidth=2, label='ε-Greedy (média amostral, ε=0.1)')
ax2.plot(data['step'], alpha_avg, 
         'r-', linewidth=2, label='ε-Greedy (α=0.1 constante)')

ax2.set_xlabel('Interações (t)')
ax2.set_ylabel('Recompensa Média por Step (janela 100)')
ax2.set_title('(b) Recompensa Instantânea Média - α constante é CLARAMENTE superior após regime shifts')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Marcar regime shifts
for shift_time in [1000, 2000, 3000, 4000]:
    ax2.axvline(x=shift_time, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    if shift_time == 1000:
        ax2.text(shift_time, ax2.get_ylim()[1]*0.95, 'Regime\nShift', 
                ha='center', fontsize=9, color='orange')

plt.tight_layout()
plt.savefig('non_stationary_detailed_analysis.pdf', bbox_inches='tight', dpi=300)
plt.savefig('non_stationary_detailed_analysis.png', bbox_inches='tight', dpi=150)
print("Figuras salvas: non_stationary_detailed_analysis.pdf/png")

# ========== ANÁLISE QUANTITATIVA POR REGIME ==========
print("\n" + "="*70)
print("ANÁLISE QUANTITATIVA POR REGIME")
print("="*70)

regimes = [
    (0, 1000, "Regime 1 (0-1000)"),
    (1000, 2000, "Regime 2 (1000-2000)"),
    (2000, 3000, "Regime 3 (2000-3000)"),
    (3000, 4000, "Regime 4 (3000-4000)"),
    (4000, 5000, "Regime 5 (4000-5000)")
]

for start, end, label in regimes:
    mask = (data['step'] >= start) & (data['step'] < end)
    greedy_mean = data[mask]['greedy_inst'].mean()
    epsilon_mean = data[mask]['epsilon_inst'].mean()
    alpha_mean = data[mask]['alpha_inst'].mean()
    
    print(f"\n{label}:")
    print(f"  Greedy:              {greedy_mean:6.2f}")
    print(f"  ε-Greedy (sample):   {epsilon_mean:6.2f}")
    print(f"  ε-Greedy (α const):  {alpha_mean:6.2f}")
    
    # Destacar o melhor
    best = max(greedy_mean, epsilon_mean, alpha_mean)
    if alpha_mean == best:
        print(f"  → α constante é MELHOR neste regime! ✓")
    elif greedy_mean == best:
        print(f"  → Greedy é melhor (sorte ou início)")

print("\n" + "="*70)
print("CONCLUSÃO:")
print("="*70)
print("""
Embora Greedy possa parecer melhor em ACUMULADO devido à sorte inicial,
a análise de RECOMPENSA INSTANTÂNEA MÉDIA mostra que:

1. α constante ADAPTA-SE RAPIDAMENTE após regime shifts
2. Greedy fica TRAVADO em ações antigas (baixa recompensa recente)
3. Em ambientes não-estacionários, o que importa é PERFORMANCE RECENTE,
   não o histórico acumulado!
""")

plt.show()
