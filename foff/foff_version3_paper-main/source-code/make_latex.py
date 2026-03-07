import numpy as np
from plotting import *
from file_utils import *
import argparse

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
data = load_data(args.filename)

def generate_latex_tables(data, scenario, queue_times=[3,6,12,25,50,100]):
    """
    Gera os códigos LaTeX para as tabelas de Energy Consumption e Total Processed Time,
    atualizando os valores para Direct (extract_nft_map) e Weighted (extract_nft_weighted).
    
    Parâmetros:
      - data: dicionário com os dados, com chaves "energy_consumption" e "processed_time".
      - scenario: string para identificar o cenário (ex: "Balanced Scenario" ou "Challenged Scenario").
      - queue_times: lista de valores de queue waiting time (ex: [3,6,12,25,50,100]).
      
    Retorna:
      Uma tupla (table_energy, table_time) contendo os códigos LaTeX para as tabelas.
      
    A estrutura esperada dos dados é:
    {
        "energy_consumption": {
            "extract_nft_map": { "3": { "FOFF": {"value": [...]}, "ROFF": {...}, "GTT": {...}, "GCF": {...} }, ... },
            "extract_nft_weighted": { "3": { "FOFF": {"value": [...]}, ... }, ... }
        },
        "processed_time": {
            "extract_nft_map": { ... },
            "extract_nft_weighted": { ... }
        }
    }
    """
    
    # Função auxiliar: calcula média e erro padrão de uma lista de valores
    def stats(values):
        arr = np.array(values)
        n = len(arr)
        if n == 0:
            return (0, 0)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0
        return (mean, se)
    
    # Define os algoritmos (colunas) na ordem desejada
    algorithms = ["FOFF", "ROFF", "GTT", "GCF"]
    
    def build_table(metric, caption, label):
        # Cabeçalho da tabela LaTeX
        header = (
            "\\begin{table*}[!h]\n"
            "    \\centering\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}}\n"
            "    \\resizebox{\\textwidth}{!}{%\n"
            "    \\begin{tabular}{c cc cc cc cc}\n"
            "        \\toprule\n"
            "        \\multirow{2}{*}{\\textbf{Queue Waiting Time}} & "
        )
        # Cria os cabeçalhos de coluna para cada algoritmo
        for algo in algorithms:
            header += f"\\multicolumn{{2}}{{c}}{{\\textbf{{{algo}}}}} & "
        header = header.rstrip(" & ") + " \\\\\n"
        header += "         & Direct & Weighted & Direct & Weighted & Direct & Weighted & Direct & Weighted \\\\\n"
        header += "        \\midrule\n"
        
        body = ""
        # Para cada valor de queue waiting time, gera uma linha da tabela
        for qt in queue_times:
            qt_key = str(qt)
            line = f"        {qt}"
            # Para cada algoritmo, calcula os valores para os dois métodos: Direct e Weighted.
            for algo in algorithms:
                # Obtém os dados para extract_nft_map (Direct)
                try:
                    values_direct = data[metric]["extract_nft_map"][qt_key][algo]["value"]
                    mean_direct, se_direct = stats(values_direct)
                except KeyError:
                    mean_direct, se_direct = (0, 0)
                # Obtém os dados para extract_nft_weighted (Weighted)
                try:
                    values_weighted = data[metric]["extract_nft_weighted"][qt_key][algo]["value"]
                    mean_weighted, se_weighted = stats(values_weighted)
                except KeyError:
                    mean_weighted, se_weighted = (0, 0)
                
                # Formata os valores com 2 casas decimais e inclui "±" para o erro
                cell_direct = f"{mean_direct:.2f} $\\pm$ {se_direct:.2f}"
                cell_weighted = f"{mean_weighted:.2f} $\\pm$ {se_weighted:.2f}"
                line += f" & {cell_direct} & {cell_weighted}"
            line += " \\\\\n"
            body += line
        
        footer = (
            "        \\bottomrule\n"
            "    \\end{tabular}%\n"
            "    }\n"
            "\\end{table*}\n"
        )
        return header + body + footer
    
    # Monta as tabelas para cada métrica
    table_energy = build_table(
        metric="energy_consumption",
        caption=f"Energy Consumption -- {scenario} (Both Methods)",
        label=f"{scenario.lower().replace(' ', '_')}_energy_combined"
    )
    table_time = build_table(
        metric="processed_time",
        caption=f"Total Processed Time (seconds) -- {scenario} (Both Methods)",
        label=f"{scenario.lower().replace(' ', '_')}_time_combined"
    )
    
    return table_energy, table_time

# Exemplo de uso:
# Suponha que 'data' seja o dicionário carregado.
scenario_str = "Balanced Scenario"  if args.scenario == 'balanced' else "Challenged Scenario"
energy_table, time_table = generate_latex_tables(data, scenario_str)
print(energy_table)
print(time_table)
