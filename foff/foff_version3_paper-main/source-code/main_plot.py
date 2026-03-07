from plotting import *
from file_utils import *
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import math

parser = argparse.ArgumentParser(description='FOFF Plot Simulator.')
parser.add_argument(
    '--filename',
    type=str,
)
parser.add_argument(
    '--scenario',
    type=str,
    choices=['challenging', 'balanced'],
    default='challenging',
    help='Cenário a ser utilizado: "challenging" ou "balanced" (default: challenging)'
)
args = parser.parse_args()
# data = load_data(args.filename)
# data = load_data('challengingdatafilename.json')
plt.rcParams['font.family'] = 'Times New Roman'

def plot_bar_with_ci(json_file, waiting_times, algorithms, metric,
                     confidence=0.95, title="Bar Chart with Confidence Intervals",
                     filename="filename", ylabel="Value", xlabel="Algorithm",
                     figsize=(11, 6), colors=None, title_size=16, label_size=14, ticks_size=12):
    # Carregar os dados do arquivo JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Para cada algoritmo, agregamos os valores da métrica para todos os waiting_times fornecidos.
    all_data = []
    for algo in algorithms:
        algo_values = []
        for wt in waiting_times:
            wt_str = str(wt)
            try:
                # Extrai a lista de valores para o algoritmo e waiting_time especificados
                values = data[metric][wt_str][algo]["value"]
            except KeyError:
                print(f"Dados não encontrados para o algoritmo '{algo}' com waiting_time {wt} na métrica '{metric}'.")
                continue
            algo_values.extend(values)
        all_data.append(np.array(algo_values))
    
    # Cálculo das médias e dos intervalos de confiança (95%) para cada algoritmo
    means = [np.mean(d) for d in all_data]
    cis = []
    for d in all_data:
        n = len(d)
        if n > 1:
            ci_val = sem(d) * t.ppf((1 + confidence) / 2, n - 1)
        else:
            ci_val = 0
        cis.append(ci_val)
    
    # Definir cores padrão, se não fornecido
    if colors is None:
        colors = ['#1f77b4', '#d6604d', '#00a693', '#db94b9', '#f0e442']
    
    # Criar o gráfico de barras com os algoritmos no eixo x
    plt.figure(figsize=figsize)
    bars = plt.bar(algorithms, means, yerr=cis, color=colors[:len(algorithms)],
                   edgecolor='black', alpha=0.7, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Adicionar rótulos acima das barras com média e intervalo de confiança
    for bar, mean_val, ci_val in zip(bars, means, cis):
        plt.text(bar.get_x() + bar.get_width() / 2, mean_val + ci_val + 0.01,
                 f'{mean_val:.2f} ± {ci_val:.2f}', ha='center', va='bottom', fontsize=ticks_size)
    
    plt.title(title, fontsize=title_size, fontweight='bold')
    plt.ylabel(ylabel, fontsize=label_size)
    plt.xlabel(xlabel, fontsize=label_size)
    # Não exibe os valores do eixo x (apenas os nomes dos algoritmos)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.show()

import matplotlib.font_manager as fm

# Check available fonts
available_fonts = [f.name for f in fm.fontManager.ttflist]
print("'Times New Roman' in available fonts:", 'Times New Roman' in available_fonts)

# Use a fallback if Times New Roman is not available
if 'Times New Roman' not in available_fonts:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif'] + plt.rcParams['font.serif']
else:
    plt.rcParams['font.family'] = 'Times New Roman'

# Exemplo de chamada da função:
waiting_times_list = [3]
# algorithms_list = ['FOFF', 'GTT', 'HOFF', 'FP-Sched', 'PSOFF', 'ROFF', 'GCF']
algorithms_list = ['FOFF', 'GTT', 'FP-Sched', 'HOFF', 'ROFF']
scenario_str = 'Challenging' if args.scenario == 'challenging' else 'Balanced'
# Para usar a métrica de consumo energético
title_size = 18.5
label_size = 16.5
ticks_size = 14.5
plot_bar_with_ci(args.filename, waiting_times_list, algorithms_list, title_size=title_size, label_size=label_size, ticks_size=ticks_size,
                 metric="energy_consumption", ylabel="Energy Consumption (Joules)", xlabel="Solution",
                 title=f"Total Energy Consumption x Solution (CI 95%)\n{scenario_str} Scenario", filename=f"{args.scenario}_energy_cost")
plot_bar_with_ci(args.filename, waiting_times_list, algorithms_list, title_size=title_size, label_size=label_size, ticks_size=ticks_size,
                 metric="processed_time", ylabel="Task Completion Time (Seconds)", xlabel="Solution",
                 title=f"Total Task Completion Time x Solution com CI 95%\n{scenario_str} Scenario", filename=f"{args.scenario}_processed_time")
plot_bar_with_ci(args.filename, waiting_times_list, algorithms_list, title_size=title_size, label_size=label_size, ticks_size=ticks_size,
                 metric="processed_cycles", ylabel="Task Completion Time (Seconds)", xlabel="Solution",
                 title=f"Total Processed Cycles x Solution com CI 95%\n{scenario_str} Scenario", filename=f"{args.scenario}_processed_cycles")