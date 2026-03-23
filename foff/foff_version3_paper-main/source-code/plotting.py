import math
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import numpy as np
import re
SHOW = False

def format_filename(text):
    # Remove characters that are not alphanumeric, spaces, or hyphens
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # Replace spaces and hyphens with underscores
    formatted_text = re.sub(r'[\s\-]+', '_', cleaned_text)
    # Convert to lowercase for consistency
    formatted_text = formatted_text.lower()
    return formatted_text

def plot_invalid_arrivals(data, algorithms, title="Invalid Arrivals Count"):
    data_sum = np.sum(data, axis=1)
    bar_width = 0.5
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, data_sum, width=bar_width, color='skyblue', edgecolor='black', alpha=0.7, capsize=5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Algorithms", fontsize=14)
    plt.ylabel("Invalid Arrivals", fontsize=14)
    plt.xticks(algorithms, fontsize=12)
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{format_filename(title)}.pdf", format="pdf")
    if SHOW == True:
        plt.show()

def plot_misses(data, algorithms, title="Misses Count"):
    data_sum = np.sum(data, axis=1)
    bar_width = 0.5
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, data_sum, width=bar_width, color='skyblue', edgecolor='black', alpha=0.7, capsize=5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Algorithms", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(algorithms, fontsize=12)
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{format_filename(title)}.pdf", format="pdf")
    if SHOW == True:
        plt.show()


def plot_device_selection_count(selection_data, methods, devices, title="Device Selection Count"):
    device_labels = [f"Edge {row[4]:02}" for row in devices]
    num_devices = len(devices)

    selection_counts = {method: {label: 0 for label in device_labels} for method in methods}

    for method, data in zip(methods, selection_data):
        for device_ids in data:  # Iterate through the lists of selected devices
            for device_id in device_ids: # Iterate through each device ID if the selection is a list
                label = f"Edge {device_id:02}"
                if label in selection_counts: #check if the label exists
                    selection_counts[label] += 1
                else:
                    print(f"Warning: Device ID {device_id} not found in device list for method {method}") #handle the case where the device id is not in the devices list

    x = np.arange(num_devices)
    bar_width = 0.2
    offset = -bar_width * (len(methods) - 1) / 2

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        counts = [selection_counts[label] for label in device_labels] #extract the counts for each device to plot them in the correct order
        plt.bar(
            x + offset + i * bar_width,
            counts,
            width=bar_width,
            label=method
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Devices", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(x, device_labels, fontsize=12, rotation=45, ha='right') #rotate x ticks for better readability if needed
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{format_filename(title)}.pdf", format="pdf")
    if SHOW == True:
        plt.show()



def plot_bar_with_ci(data, algorithms, confidence=0.95, title="Bar Chart with Confidence Intervals", filename="filename.pdf",
                     ylabel="Value", xlabel="Algorithm", figsize=(11, 6), colors=None):
    """
    Plota um gráfico de barras com intervalos de confiança.
    
    Parâmetros:
        - data (np.array): Matriz de resultados, com cada linha representando um algoritmo.
        - algorithms (list): Lista de nomes dos algoritmos correspondentes às linhas de `data`.
        - confidence (float): Nível de confiança para o intervalo de confiança (default: 95%).
        - title (str): Título do gráfico.
        - ylabel (str): Rótulo do eixo Y.
        - xlabel (str): Rótulo do eixo X.
        - figsize (tuple): Tamanho da figura do gráfico.
        - colors (list): Lista de cores para as barras.
    """

    # Converte `data` para NumPy array caso ainda não seja
    data = np.array(data)

    # Garantir que `data` tenha pelo menos 2 dimensões (caso seja 1D)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Verificar o número de algoritmos e se bate com os dados
    if len(algorithms) != data.shape[0]:
        raise ValueError(f"Erro: Número de algoritmos ({len(algorithms)}) não corresponde ao número de linhas de dados ({data.shape[0]}).")

    # Calcular médias e intervalos de confiança
    means = np.mean(data, axis=1)
    n = data.shape[1]  # Número de execuções por estratégia

    if n > 1:
        ci_error = sem(data, axis=1) * t.ppf((1 + confidence) / 2, n - 1)  # IC para n > 1
    else:
        ci_error = np.zeros_like(means)  # Se `n = 1`, IC = 0

    # Paleta de cores padrão
    if colors is None:
        colors = ['#1f77b4', '#d6604d', '#00a693', '#db94b9', '#f0e442', '#7570b3', '#e7298a', '#ff7f00']

    # Criar a figura
    plt.figure(figsize=figsize)
    bars = plt.bar(algorithms, means, yerr=ci_error, color=colors[:len(algorithms)], edgecolor='black',
                   alpha=0.7, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

    # Adicionar rótulos acima das barras
    for bar, ci in zip(bars, ci_error):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + ci + 0.01,
                 f'{yval:.2f} ± {ci:.2f}', ha='center', va='bottom', fontsize=12)

    # Configuração dos rótulos, título e grid
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{format_filename(filename)}.pdf", format="pdf")
    # Exibir o gráfico
    if SHOW == True:
        plt.show()


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_relative_differences(data, scenario):   
    def get_quadrant_color(x, y):
        if x < 0 and y > 0:
            return "#7570b3"
        elif x > 0 and y > 0:
            return "#00a693"
        elif x < 0 and y < 0:
            return "#db94b9"
        else:  # x > 0 and y < 0
            return "#d6604d"
    
    def get_diff(extract_method, metric, t):
        # Usa o primeiro valor (índice 0) para o cálculo
        FOFF_values = data[metric][extract_method][t]["FOFF"]["value"]
        GTT_values  = data[metric][extract_method][t]["GTT"]["value"]
        # Calcula a média dos valores
        FOFF_val = sum(FOFF_values) / len(FOFF_values)
        GTT_val  = sum(GTT_values) / len(GTT_values)
        return 1.0 - (FOFF_val / GTT_val) if GTT_val != 0 else 0.0

    # Obtém os parâmetros (chaves) que aparecem em ambas as abordagens
    wt_weighted = set(data["energy_consumption"]["extract_nft_weighted"].keys())
    wt_mapping = set(data["energy_consumption"]["extract_nft_map"].keys())
    queue_times = sorted(wt_weighted & wt_mapping, key=lambda x: float(x))
    
    # Vetores para armazenar as diferenças relativas para cada método
    energy_rel_weighted = []
    latency_rel_weighted = []
    energy_rel_mapping = []
    latency_rel_mapping = []
    
    for t in queue_times:
        energy_rel_weighted.append(get_diff("extract_nft_weighted", "energy_consumption", t))
        latency_rel_weighted.append(get_diff("extract_nft_weighted", "processed_time", t))
        energy_rel_mapping.append(get_diff("extract_nft_map", "energy_consumption", t))
        latency_rel_mapping.append(get_diff("extract_nft_map", "processed_time", t))
    
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    
    # Plot para o método weighted (círculos)
    for i, t in enumerate(queue_times):
        x = energy_rel_weighted[i]
        y = latency_rel_weighted[i]
        color = get_quadrant_color(x, y)
        plt.scatter(x, y, marker='o', color=color)
        plt.annotate(str(t), (x, y), textcoords="offset points", xytext=(0,5),
                     ha='center', fontsize=9, color='#1A1A1A')
    
    # Plot para o método mapping (quadrados)
    for i, t in enumerate(queue_times):
        x = energy_rel_mapping[i]
        y = latency_rel_mapping[i]
        color = get_quadrant_color(x, y)
        plt.scatter(x, y, marker='s', color=color)
        plt.annotate(str(t), (x, y), textcoords="offset points", xytext=(0,5),
                     ha='center', fontsize=9, color='#1A1A1A')
    
    # Linhas de referência nos eixos
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.xlabel("Relative Difference in Energy Cost (FOFF vs GTT)", fontsize=14)
    plt.ylabel("Relative Difference in Latency (FOFF vs GTT)", fontsize=14)
    title = f"Relative Difference of FOFF Metrics Compared to GTT\n{scenario} Scenario"
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Legendas customizadas
    legend_handles = [
        mlines.Line2D([], [], color="#7570b3", marker='o', linestyle='None',
                      label="Weighted - Better Energy & Worse Latency"),
        mlines.Line2D([], [], color="#00a693", marker='o', linestyle='None',
                      label="Weighted - Better Energy & Latency"),
        mlines.Line2D([], [], color="#db94b9", marker='o', linestyle='None',
                      label="Weighted - Worse Energy & Latency"),
        mlines.Line2D([], [], color="#d6604d", marker='o', linestyle='None',
                      label="Weighted - Worse Energy & Better Latency"),
        mlines.Line2D([], [], color="#7570b3", marker='s', linestyle='None',
                      label="Mapping - Better Energy & Worse Latency"),
        mlines.Line2D([], [], color="#00a693", marker='s', linestyle='None',
                      label="Mapping - Better Energy & Latency"),
        mlines.Line2D([], [], color="#db94b9", marker='s', linestyle='None',
                      label="Mapping - Worse Energy & Latency"),
        mlines.Line2D([], [], color="#d6604d", marker='s', linestyle='None',
                      label="Mapping - Worse Energy & Better Latency"),
    ]
    
    plt.legend(handles=legend_handles, fontsize=9, loc='best')
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_line_comparison(data, metric, queue_time, algorithms):
    # Converte o valor do queue time para string
    qt_key = str(queue_time)     
    plt.figure(figsize=(12, 6))   

    for algo in algorithms:
        if algo not in data[metric][qt_key]:
            print(f"Algoritmo {algo} não encontrado com queue time {qt_key}.")
            continue
        series = data[metric][qt_key][algo]["value"]
        x = range(len(series))
        plt.plot(x, series, marker='o', label=f"{algo}")
    
    plt.xlabel("Índice dos Valores")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Comparação de {metric.replace('_', ' ')} para queue waiting time {qt_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# plot_line_comparison(data, metric="energy_consumption", queue_time=3, algorithms=["FOFF", "ROFF", "GTT", "GCF"])

def plot_line_with_confidence(data, metric, queue_times, algorithms):   
    plt.figure(figsize=(12, 6))    
    # Para cada método e para cada algoritmo, calcula a média e o intervalo de confiança de 95% em cada queue time.

    for algo in algorithms:
        means = []
        lower = []
        upper = []
        x_vals = []
        for qt in queue_times:
            qt_key = str(qt)
            # Verifica se o queue time existe para este método e métrica
            if qt_key not in data[metric]:
                print(f"Queue waiting time {qt_key} não encontrado.")
                continue
            if algo not in data[metric][qt_key]:
                print(f"Algoritmo {algo} não encontrado para o queue time {qt_key}.")
                continue
            values = data[metric][qt_key][algo]["value"]
            arr = np.array(values)
            n = len(arr)
            if n == 0:
                continue
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)  # desvio padrão amostral
            ci = 1.96 * (std_val / np.sqrt(n))
            means.append(mean_val)
            lower.append(mean_val - ci)
            upper.append(mean_val + ci)
            x_vals.append(qt)
        
        if x_vals:  # Se houver dados, plota a linha e a banda de confiança.
            plt.plot(x_vals, means, marker='o', label=f"{algo}")
            plt.fill_between(x_vals, lower, upper, alpha=0.2)
    plt.xlabel("Queue Waiting Time")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} com Intervalo de Confiança de 95%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(queue_times)
    # plt.yscale('log')
    plt.show()
# plot_line_with_confidence(data, metric="energy_consumption", queue_times=[3, 6, 12, 25, 50, 100], algorithms=["FOFF", "ROFF", "GTT", "GCF"])

def plot_line_with_confidence_selected(data, metric, queue_times, algorithms, method):
    plt.figure(figsize=(12, 6))
    
    # Itera sobre cada algoritmo
    for algo in algorithms:
        means = []
        lower = []
        upper = []
        x_vals = []
        
        for qt in queue_times:
            qt_key = str(qt)
            # Verifica se o queue time existe para o método e métrica escolhidos
            if qt_key not in data[metric]:
                print(f"Queue waiting time {qt_key} não encontrado para o método {method}.")
                continue
            if algo not in data[metric][qt_key]:
                print(f"Algoritmo {algo} não encontrado para o método {method} com queue time {qt_key}.")
                continue
            
            values = data[metric][qt_key][algo]["value"]
            arr = np.array(values)
            n = len(arr)
            if n == 0:
                continue
            
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)  # desvio padrão amostral
            ci = 1.96 * (std_val / np.sqrt(n))
            
            means.append(mean_val)
            lower.append(mean_val - ci)
            upper.append(mean_val + ci)
            x_vals.append(qt)
        
        if x_vals:
            plt.plot(x_vals, means, marker='o', label=f"{method} - {algo}")
            plt.fill_between(x_vals, lower, upper, alpha=0.2)
    
    plt.xlabel("Queue Waiting Time")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} com Intervalo de Confiança de 95%\nMétodo: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.yscale('log')
    plt.xticks(queue_times)
    plt.show()

# Exemplo de uso:
# plot_line_with_confidence_selected(data, metric="energy_consumption", queue_times=[3, 6, 12, 25, 50, 100],
#                                    algorithms=["FOFF", "ROFF", "GTT", "GCF"], method="extract_nft_map")

def plot_graph_with_confidence_custom(data, metric, queue_times, algorithm_list, method_list):
    if len(algorithm_list) != len(method_list):
        print("A lista de algoritmos e a lista de métodos devem ter o mesmo tamanho.")
        return

    plt.figure(figsize=(12, 6))
    
    # Para cada par (algoritmo, método) fornecido, calcula a média e o intervalo de confiança em cada queue time
    for algo, method in zip(algorithm_list, method_list):
        means = []
        lower = []
        upper = []
        x_vals = []
        
        for qt in queue_times:
            qt_key = str(qt)
            # Verifica se o queue time existe para o método e métrica escolhidos
            if qt_key not in data[metric]:
                print(f"Queue waiting time {qt_key} não encontrado para o método {method}.")
                continue
            if algo not in data[metric][qt_key]:
                print(f"Algoritmo {algo} não encontrado para o método {method} com queue time {qt_key}.")
                continue
            
            values = data[metric][qt_key][algo]["value"]
            arr = np.array(values)
            n = len(arr)
            if n == 0:
                continue
            
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)  # desvio padrão amostral
            ci = 1.96 * (std_val / np.sqrt(n))
            
            means.append(mean_val)
            lower.append(mean_val - ci)
            upper.append(mean_val + ci)
            x_vals.append(qt)
        
        # Se houver dados para esse par, plota a linha e a banda de confiança
        if x_vals:
            plt.plot(x_vals, means, marker='o', label=f"{method} - {algo}")
            plt.fill_between(x_vals, lower, upper, alpha=0.2)
    
    plt.xlabel("Queue Waiting Time")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} com Intervalo de Confiança de 95%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(queue_times)
    plt.show()

# Exemplo de uso:
# plot_graph_with_confidence_custom(data, metric="energy_consumption",
#                                   queue_times=[3, 6, 12, 25, 50, 100],
#                                   algorithm_list=["FOFF", "FOFF", "GTT"],
#                                   method_list=["extract_nft_map", "extract_nft_weighted", "extract_nft_map"])

def plot_improvement_over_last(data, metric, queue_times, algorithm_list, method_list):
    if len(algorithm_list) < 3:
        return
    last_algorithm = algorithm_list[-1]
    last_method = method_list[-1]
    # Verifica se as listas possuem o mesmo tamanho
    if len(algorithm_list) != len(method_list):
        print("Erro: algorithm_list e method_list devem ter o mesmo tamanho.")
        return

    # Primeiro, coletamos os dados baseline de GTT (usando extract_nft_map) para cada queue time.
    gtt_stats = {}
    for qt in queue_times:
        qt_key = str(qt)
        if qt_key not in data[metric]["extract_nft_map"]:
            print(f"Queue time {qt_key} não encontrado em extract_nft_map para GTT.")
            continue
        if last_algorithm not in data[metric]["extract_nft_map"][qt_key]:
            print(f"GTT não encontrado em extract_nft_map para queue time {qt_key}.")
            continue
        vals = np.array(data[metric]["extract_nft_map"][qt_key][last_algorithm]["value"])
        if len(vals) == 0:
            continue
        mean_gtt = np.mean(vals)
        se_gtt = np.std(vals, ddof=1) / math.sqrt(len(vals))
        gtt_stats[qt_key] = (mean_gtt, se_gtt)
    
    # Para cada par (algoritmo, método) da entrada, calculamos a melhoria relativa e seu intervalo de confiança.
    results = []
    for algo, method in zip(algorithm_list, method_list):
        results.append({
            "label": f"{method} - {algo}",
            "x": [],
            "mean": [],
            "lower": [],
            "upper": []
        })
    
    # Itera por cada queue time e calcula os valores
    for qt in queue_times:
        qt_key = str(qt)
        if qt_key not in gtt_stats:
            continue
        m_gtt, se_gtt = gtt_stats[qt_key]
        for idx, (algo, method) in enumerate(zip(algorithm_list, method_list)):
            results[idx]["x"].append(qt)
            # Se o algoritmo for GTT, consideramos melhoria = 0 (baseline)
            if algo.upper() == last_algorithm:
                results[idx]["mean"].append(0)
                results[idx]["lower"].append(0)
                results[idx]["upper"].append(0)
                continue
            # Caso contrário, esperamos que seja FOFF
            if qt_key not in data[metric]:
                print(f"Queue time {qt_key} não encontrado para o método {method}.")
                continue
            if algo not in data[metric][qt_key]:
                print(f"Algoritmo {algo} não encontrado para o método {method} com queue time {qt_key}.")
                continue
            values = np.array(data[metric][qt_key][algo]["value"])
            if len(values) == 0:
                continue
            m_foff = np.mean(values)
            se_foff = np.std(values, ddof=1) / math.sqrt(len(values))
            # Melhoria relativa
            improvement = 1 - (m_foff / m_gtt)
            # Estima a variância de (m_foff/m_gtt) pelo método delta:
            var_ratio = (se_foff / m_gtt)**2 + ((m_foff * se_gtt) / (m_gtt**2))**2
            se_improvement = math.sqrt(var_ratio)
            ci = 1.96 * se_improvement
            results[idx]["mean"].append(improvement)
            results[idx]["lower"].append(improvement - ci)
            results[idx]["upper"].append(improvement + ci)
    
    # Plotagem
    plt.figure(figsize=(12, 6))
    for res in results:
        if len(res["x"]) > 0:
            plt.plot(res["x"], res["mean"], marker="o", label=res["label"])
            plt.fill_between(res["x"], res["lower"], res["upper"], alpha=0.2)
    plt.xlabel("Queue Waiting Time")
    plt.ylabel("Melhoria Relativa (1 - FOFF / GTT)")
    plt.title(f"{metric.replace('_',' ').title()} - Melhoria de FOFF em Relação a GTT (IC 95%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(queue_times)
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# plot_improvement_over_GTT(data,
#                           metric="energy_consumption",
#                           queue_times=[3, 6, 12, 25, 50, 100],
#                           algorithm_list=["FOFF", "FOFF", "GTT"],
#                           method_list=["extract_nft_map", "extract_nft_weighted", "extract_nft_map"])