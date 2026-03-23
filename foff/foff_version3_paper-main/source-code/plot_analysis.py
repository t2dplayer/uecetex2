"""
FOFF Visualization Script

Este script carrega os resultados salvos da análise de sensibilidade do Fuzzy Offloading Framework (FOFF)
e gera visualizações sem executar novamente as simulações.
Inclui controle centralizado de tamanhos de fonte para todos os elementos gráficos.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json
from collections import defaultdict
import matplotlib as mpl

# Certifique-se de que o diretório para salvar os gráficos existe
os.makedirs("sensitivity_results", exist_ok=True)

# Classe para configuração global de estilos dos gráficos
class PlotConfig:
    """Configuração centralizada para os estilos de gráficos"""
    
    def __init__(self, scale_factor=1.0):
        """
        Inicializa a configuração de plotagem com um fator de escala
        
        Args:
            scale_factor: Fator multiplicador para todos os tamanhos de fonte (padrão: 1.0)
        """
        self.scale_factor = scale_factor
        
        # Tamanhos de fonte base para diferentes elementos
        self._base_title_size = 16
        self._base_label_size = 14
        self._base_tick_size = 12
        self._base_legend_size = 12
        self._base_annotation_size = 10
        
        # Aplica configurações ao matplotlib
        self.apply_settings()
    
    @property
    def title_size(self):
        """Tamanho da fonte de títulos"""
        return int(self._base_title_size * self.scale_factor)
    
    @property
    def label_size(self):
        """Tamanho da fonte de rótulos de eixos"""
        return int(self._base_label_size * self.scale_factor)
    
    @property
    def tick_size(self):
        """Tamanho da fonte de marcações de eixos"""
        return int(self._base_tick_size * self.scale_factor)
    
    @property
    def legend_size(self):
        """Tamanho da fonte da legenda"""
        return int(self._base_legend_size * self.scale_factor)
    
    @property
    def annotation_size(self):
        """Tamanho da fonte para anotações nos gráficos"""
        return int(self._base_annotation_size * self.scale_factor)
    
    def update_scale(self, new_scale):
        """Atualiza o fator de escala e reaplica as configurações"""
        self.scale_factor = new_scale
        self.apply_settings()
    
    def apply_settings(self):
        """Aplica as configurações ao matplotlib"""
        # Configurações gerais do matplotlib
        plt.rcParams['font.size'] = self.tick_size
        plt.rcParams['axes.titlesize'] = self.title_size
        plt.rcParams['axes.labelsize'] = self.label_size
        plt.rcParams['xtick.labelsize'] = self.tick_size
        plt.rcParams['ytick.labelsize'] = self.tick_size
        plt.rcParams['legend.fontsize'] = self.legend_size
        plt.rcParams['figure.titlesize'] = self.title_size + 2
        
        # Configurações para o Seaborn
        sns.set(font_scale=self.scale_factor)

# Instancia a configuração global com o fator de escala padrão
# Este valor pode ser modificado facilmente para ajustar todos os tamanhos de fonte
PLOT_CONFIG = PlotConfig(scale_factor=2.0)

# Função para carregar os resultados salvos
def load_results(filename):
    """Carrega os resultados de um arquivo JSON"""
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro ao decodificar o arquivo JSON {filename}.")
        return None

class FOFFVisualizer:
    """Classe para visualizar os resultados da análise de sensibilidade do FOFF"""
    
    def __init__(self):
        """Inicializa o visualizador"""
        # Níveis de importância para pesos fuzzy
        self.importances = ["vl", "l", "ml", "m", "mh", "h", "vh"]
    
    def create_summary_dataframe(self, results):
        """Cria um dataframe resumo a partir dos resultados"""
        
        # Estrutura para armazenar os dados resumidos
        summary_data = []
        
        # Processa estratégias baseline primeiro
        if "Baseline" in results:
            for strategy_name, data in results["Baseline"].items():
                summary_data.append({
                    "Strategy": strategy_name,
                    "Omega": "N/A",
                    "Energy (J)": np.mean(data["metrics"]["energy"]),
                    "Energy Std": np.std(data["metrics"]["energy"]),
                    "Latency (s)": np.mean(data["metrics"]["latency"]),
                    "Latency Std": np.std(data["metrics"]["latency"]),
                    "Decision Time (s)": np.mean(data["metrics"]["decision_times"]),
                    "Energy Improvement (%)": 0.0,
                    "Latency Improvement (%)": 0.0,
                    "Average Improvement (%)": 0.0
                })
        
        # Processa estratégia FOFF com diferentes valores de omega
        if "FOFF" in results:
            for strategy_key, data in results["FOFF"].items():
                omega_str = "N/A"
                if "omega" in data:
                    omega_str = f"{data['omega'][0]},{data['omega'][1]}"
                
                summary_data.append({
                    "Strategy": data.get("strategy", "FOFF"),
                    "Omega": omega_str,
                    "Energy (J)": np.mean(data["metrics"]["energy"]),
                    "Energy Std": np.std(data["metrics"]["energy"]),
                    "Latency (s)": np.mean(data["metrics"]["latency"]),
                    "Latency Std": np.std(data["metrics"]["latency"]),
                    "Decision Time (s)": np.mean(data["metrics"]["decision_times"]),
                    "Energy Improvement (%)": data["improvements"]["energy_improvement"] if "improvements" in data else 0.0,
                    "Latency Improvement (%)": data["improvements"]["latency_improvement"] if "improvements" in data else 0.0,
                    "Average Improvement (%)": ((data["improvements"]["energy_improvement"] + 
                                               data["improvements"]["latency_improvement"]) / 2) if "improvements" in data else 0.0
                })
        
        return pd.DataFrame(summary_data)
    
    def analyze_results(self, results):
        """Analisa os resultados de sensibilidade e gera visualizações"""
        
        # Cria dataframe de resumo
        df = self.create_summary_dataframe(results)
        
        # Salva resumo para CSV
        df.to_csv("sensitivity_results/sensitivity_summary.csv", index=False)
        
        # Cria visualizações
        self._plot_heatmap_analysis(results)
        self._plot_best_configurations(df)
        self._analyze_device_selection(results)
        
        # Gráfico de análise de tradeoff energia-latência
        self._create_energy_latency_tradeoff_table(df)        
        print("Visualizações geradas com sucesso no diretório 'sensitivity_results'.")
        
        return df
    
    def _prepare_heatmap_data(self, strategy_results, metric):
        """Prepara dados para visualização de heatmap"""
        
        # Cria dataframe com importâncias como índices e colunas
        heatmap_data = pd.DataFrame(index=self.importances, columns=self.importances)
        
        # Preenche com zeros inicialmente para evitar valores NaN
        for idx in self.importances:
            for col in self.importances:
                heatmap_data.loc[idx, col] = 0.0
        
        for strategy_key, data in strategy_results.items():
            # Extrai pesos de omega
            if "omega" in data:
                w1, w2 = data["omega"]
                
                # Preenche dados do heatmap
                if "improvements" in data and metric in data["improvements"]:
                    # Garante que o valor é um float
                    try:
                        value = float(data["improvements"][metric])
                        heatmap_data.loc[w1, w2] = value
                    except (ValueError, TypeError):
                        # Se a conversão falhar, usa 0.0
                        heatmap_data.loc[w1, w2] = 0.0
        
        # Converte explicitamente todos os dados para float
        heatmap_data = heatmap_data.astype(float)
        
        return heatmap_data
    
    def _plot_heatmap_analysis(self, results):
        """Gera heatmaps para visualizar sensibilidade de parâmetros"""
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif'] + plt.rcParams['font.serif']        
        try:
            # Envolve cada operação de plot em try-except para garantir que uma falha
            # não pare todos os plots
            try:
                # Prepara dados para heatmap de eficiência energética
                energy_data_foff = self._prepare_heatmap_data(results["FOFF"], "energy_improvement")
                
                # Plot heatmap de eficiência energética para FOFF
                plt.figure(figsize=(10, 8))
                sns.heatmap(energy_data_foff, annot=True, cmap="RdYlGn", fmt=".1f")
                plt.title("Energy Efficiency Improvements (%)", fontweight='bold', fontsize=PLOT_CONFIG.label_size)
                plt.xlabel("Energy Weight ($\omega_2$)")
                plt.ylabel("Latency Weight ($\omega_1$)")
                plt.tight_layout()
                plt.savefig("sensitivity_results/energy_heatmap_FOFF.pdf", format="pdf")
                plt.close()
            except Exception as e:
                print(f"Erro ao plotar o heatmap de energia FOFF: {e}")
            
            try:
                # Prepara dados para heatmap de melhoria de latência
                latency_data_foff = self._prepare_heatmap_data(results["FOFF"], "latency_improvement")
                
                # Plot heatmap de melhoria de latência para FOFF
                plt.figure(figsize=(10, 8))
                sns.heatmap(latency_data_foff, annot=True, cmap="RdYlGn", fmt=".1f")
                plt.title("Task Completion Time Improvements (%)", fontweight='bold', fontsize=PLOT_CONFIG.label_size)
                plt.xlabel("Energy Weight ($\omega_2$)")
                plt.ylabel("Latency Weight ($\omega_1$)")
                plt.tight_layout()
                plt.savefig("sensitivity_results/latency_heatmap_FOFF.pdf", format="pdf")
                plt.close()
            except Exception as e:
                print(f"Erro ao plotar o heatmap de latência FOFF: {e}")
                
        except Exception as e:
            print(f"Erro na análise de heatmap: {e}")
            print("Continuando com outras análises...")
    
    def _plot_best_configurations(self, df):
        """Plota as melhores configurações de parâmetros para diferentes valores de omega"""
        
        # Filtra estratégias FOFF
        foff_df = df[df["Strategy"] == "FOFF"]
        
        # Classifica por melhoria média para encontrar as melhores configurações
        best_configs = foff_df.sort_values("Average Improvement (%)", ascending=False).head(5)
        
        # Plota comparação das melhores configurações de omega
        plt.figure(figsize=(14, 10))
        
        # Cria gráfico de barras agrupadas
        x = np.arange(len(best_configs))
        width = 0.35
        
        plt.bar(x - width/2, best_configs["Energy Improvement (%)"], width, 
               label="Energy Improvement (%)")
        plt.bar(x + width/2, best_configs["Latency Improvement (%)"], width, 
               label="Latency Improvement (%)")
        
        plt.xlabel("FOFF Weight Configurations (ω)")
        plt.ylabel("Improvement Percentage (%)")
        plt.title("Best FOFF Parameter Configurations", fontweight='bold', fontsize=PLOT_CONFIG.label_size)
        
        # Cria rótulos do eixo x com valores de omega
        labels = [f"ω={row['Omega']}" for _, row in best_configs.iterrows()]
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=PLOT_CONFIG.tick_size)
        
        plt.legend(fontsize=PLOT_CONFIG.legend_size)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Adiciona rótulos de valor
        for i, value in enumerate(best_configs["Energy Improvement (%)"]):
            plt.text(i - width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        for i, value in enumerate(best_configs["Latency Improvement (%)"]):
            plt.text(i + width/2, value + 0.5, f"{value:.1f}%", ha='center')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif'] + plt.rcParams['font.serif']
        plt.savefig("sensitivity_results/best_configurations.pdf", format="pdf")
        plt.close()
        
        # Salva as melhores configurações para CSV
        best_configs.to_csv("sensitivity_results/best_configurations.csv", index=False)
        
        # Também plota as piores configurações para comparação
        worst_configs = foff_df.sort_values("Average Improvement (%)", ascending=True).head(5)
        
        plt.figure(figsize=(14, 10))
        x = np.arange(len(worst_configs))
        
        plt.bar(x - width/2, worst_configs["Energy Improvement (%)"], width, 
               label="Energy Improvement (%)")
        plt.bar(x + width/2, worst_configs["Latency Improvement (%)"], width, 
               label="Latency Improvement (%)")
        
        plt.xlabel("FOFF Weight Configurations (ω)")
        plt.ylabel("Improvement Percentage (%)")
        plt.title("Worst FOFF Parameter Configurations", fontweight='bold', fontsize=PLOT_CONFIG.label_size)
        
        labels = [f"ω={row['Omega']}" for _, row in worst_configs.iterrows()]
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=PLOT_CONFIG.tick_size)
        
        plt.legend(fontsize=PLOT_CONFIG.legend_size)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        for i, value in enumerate(worst_configs["Energy Improvement (%)"]):
            plt.text(i - width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        for i, value in enumerate(worst_configs["Latency Improvement (%)"]):
            plt.text(i + width/2, value + 0.5, f"{value:.1f}%", ha='center')
        plt.savefig("sensitivity_results/worst_configurations.pdf", format="pdf")
        plt.close()
        
    def _analyze_device_selection(self, results):
        """Analisa e visualiza padrões de seleção de dispositivos"""
        
        # Extrai dados de seleção de dispositivos
        selection_data = defaultdict(list)
        
        # Processa estratégias de baseline
        for strategy_name, data in results["Baseline"].items():
            key = f"{strategy_name}"
            
            # Conta seleções de dispositivos
            device_counts = defaultdict(int)
            
            for selections in data["metrics"]["selections"]:
                for device_id in selections:
                    device_counts[device_id] += 1
            
            selection_data[key].append(device_counts)
                
        # Processa estratégia FOFF com diferentes valores de omega
        for strategy_key, data in results["FOFF"].items():
            if "omega" in data:
                key = f"FOFF-{data['omega'][0]}_{data['omega'][1]}"
                
                # Conta seleções de dispositivos
                device_counts = defaultdict(int)
                
                for selections in data["metrics"]["selections"]:
                    for device_id in selections:
                        device_counts[device_id] += 1
                
                selection_data[key].append(device_counts)
        
        # Cria uma visualização composta de padrão de seleção
        plt.figure(figsize=(15, 10))
        
        # Identifica configurações chave para visualizar
        key_configs = [
            # Estratégias de baseline
            # "Random", "GTT", "GCF",
            # FOFF com diferentes combinações de pesos
            "FOFF-vh_vl",  # Focado em energia
            "FOFF-vl_vh",  # Focado em latência
            "FOFF-m_m",    # Balanceado
            "FOFF-h_l",    
            "FOFF-l_h",
            "FOFF-ml_mh"
        ]
        
        # Filtra para incluir apenas configurações que existem nos resultados
        available_configs = [config for config in key_configs if config in selection_data]
        
        # Prepara dados para o plot
        num_configs = len(available_configs)
        num_devices = 6  # Assumindo 6 dispositivos (0-5)
        bar_width = 0.8 / num_configs
        
        # Cria gráfico de barras agrupadas de porcentagens de seleção de dispositivos
        for i, config in enumerate(available_configs):
            # Agrega contagens de dispositivos
            aggregated_counts = defaultdict(int)
            for counts in selection_data[config]:
                for device_id, count in counts.items():
                    aggregated_counts[device_id] += count
            
            # Converte para porcentagens
            total = sum(aggregated_counts.values())
            if total > 0:
                percentages = [
                    (aggregated_counts.get(device_id, 0) / total) * 100
                    for device_id in range(num_devices)
                ]
                
                # Calcula posições x para barras
                x = np.arange(num_devices)
                offset = (i - num_configs / 2) * bar_width + bar_width / 2
                
                # Plota barras
                plt.bar(x + offset, percentages, bar_width, label=config)
        
        plt.xlabel("Device ID", fontsize=PLOT_CONFIG.label_size)
        plt.ylabel("Selection Percentage (%)", fontsize=PLOT_CONFIG.label_size)
        plt.title("Device Selection Patterns for Different Algorithms and FOFF Weight Configurations", 
                 fontsize=PLOT_CONFIG.title_size)
        
        plt.xticks(np.arange(num_devices), [str(i) for i in range(num_devices)], 
                  fontsize=PLOT_CONFIG.tick_size)
        plt.yticks(fontsize=PLOT_CONFIG.tick_size)
        
        # Configure o tamanho da legenda e sua posição
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=3, fontsize=PLOT_CONFIG.legend_size)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif'] + plt.rcParams['font.serif']
        plt.savefig("sensitivity_results/device_selection_patterns.pdf", format="pdf")
        plt.close()
    
    def _create_energy_latency_tradeoff_table(self, df):
        """
        Cria uma tabela LaTeX que mostra a relação de tradeoff entre melhoria de 
        energia e latência para diferentes configurações de pesos do FOFF.
        
        Args:
            df: DataFrame com os dados de resultados
            
        Returns:
            String contendo código LaTeX para a tabela
        """
        # Filtra para incluir apenas estratégias FOFF
        foff_df = df[df["Strategy"] == "FOFF"]
        
        # Ordena o dataframe por melhoria de energia (decrescente)
        energy_sorted = foff_df.sort_values("Energy Improvement (%)", ascending=False)
        
        # Inicializa a string LaTeX
        latex_table = """
    \\begin{table}[htbp]
    \\centering
    \\caption{Melhorias de Energia e Tempo de Conclusão para Diferentes Configurações de Pesos FOFF}
    \\label{tab:energy_latency_tradeoff}
    \\begin{tabular}{|c|c|c|c|c|}
    \\hline
    \\textbf{ID} & \\textbf{Configuração de Pesos ($\\omega$)} & \\textbf{Melhoria de Energia (\\%)} & \\textbf{Melhoria de Tempo (\\%)} & \\textbf{Melhoria Média (\\%)} \\\\
    \\hline
    """
        
        # Adiciona linhas para cada configuração
        for idx, row in enumerate(energy_sorted.iterrows(), 1):
            _, data = row
            omega = data["Omega"].replace(",", ", ")
            energy_improvement = f"{data['Energy Improvement (%)']:.2f}"
            latency_improvement = f"{data['Latency Improvement (%)']:.2f}"
            avg_improvement = f"{data['Average Improvement (%)']:.2f}"
            
            latex_table += f"{idx} & ({omega}) & {energy_improvement} & {latency_improvement} & {avg_improvement} \\\\\n\\hline\n"
        
        # Fecha a tabela
        latex_table += """\\end{tabular}
    \\end{table}
    """
        
        # Imprime a tabela LaTeX no console para fácil cópia
        print("Tabela LaTeX de Trade-off Energia-Latência:")
        print(latex_table)
        
        # Salva a tabela em um arquivo .tex
        with open("sensitivity_results/energy_latency_tradeoff_table.tex", "w") as f:
            f.write(latex_table)
        
        print("Tabela LaTeX salva em: sensitivity_results/energy_latency_tradeoff_table.tex")
        
        return latex_table
# Função principal de execução
def main():
    # Caminho para o arquivo de resultados
    results_file = "sensitivity_results/raw_results.json"
    
    # Define o fator de escala para todos os tamanhos de fonte
    # Modifique este valor para aumentar ou diminuir todos os tamanhos de fonte
    # 1.0 = tamanho normal, 1.5 = 50% maior, 0.8 = 20% menor, etc.
    font_scale = 1.8  # Exemplo: aumenta as fontes em 20%
    
    # Atualiza a configuração de plotagem com o novo fator de escala
    PLOT_CONFIG.update_scale(font_scale)
    
    print(f"Configuração de fontes: fator de escala = {font_scale}")
    print(f"  Título: {PLOT_CONFIG.title_size}pt")
    print(f"  Rótulos: {PLOT_CONFIG.label_size}pt")
    print(f"  Marcadores: {PLOT_CONFIG.tick_size}pt")
    print(f"  Legenda: {PLOT_CONFIG.legend_size}pt")
    print(f"  Anotações: {PLOT_CONFIG.annotation_size}pt")
    
    # Carrega os resultados
    results = load_results(results_file)
    
    if results:
        print(f"Resultados carregados com sucesso do arquivo {results_file}.")
        
        # Cria visualizador e gera visualizações
        visualizer = FOFFVisualizer()
        summary_df = visualizer.analyze_results(results)
        
        # Imprime resumo das melhores configurações
        best_configs = summary_df[summary_df["Strategy"] == "FOFF"].sort_values(
            "Average Improvement (%)", ascending=False).head(5)
        
        print("\nMelhores Configurações de Parâmetros FOFF:")
        for idx, row in best_configs.iterrows():
            print(f"  Omega: {row['Omega']}")
            print(f"    Melhoria de Energia: {row['Energy Improvement (%)']:.2f}%")
            print(f"    Melhoria de Latência: {row['Latency Improvement (%)']:.2f}%")
            print(f"    Melhoria Média: {row['Average Improvement (%)']:.2f}%")
        
        print("\nVisualização completa. Resultados salvos no diretório 'sensitivity_results'.")
    else:
        print("Não foi possível carregar os resultados. Certifique-se de que o arquivo existe.")
        print("Verifique se você executou a análise de sensibilidade anteriormente.")

if __name__ == "__main__":

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
    main()