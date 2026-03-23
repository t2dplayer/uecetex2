import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from dataclasses import dataclass
import pandas as pd
from typing import List, Tuple, Dict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SimulationConfig
from device_manager import Device
from task_manager import Task
from wireless_5g import *
from energy_consumption import *


# Configuração da simulação
def create_simulation_setup():
    # Parâmetros de rede sem fio 5G
    wireless_params_los = WirelessParameters(
        frequency_ghz=3.5,  # Frequência 5G típica (3.5 GHz)
        bandwidth_hz=100e6,  # 100 MHz de largura de banda
        tx_power_dbm=23.0,   # 23 dBm ~= 200 mW
        antenna_height=10.0,  # 10 metros
        noise_density_dbm_hz=-174.0,  # Densidade de ruído térmico
        los_type='los'  # Line of sight
    )
    
    wireless_params_nlos = WirelessParameters(
        frequency_ghz=3.5,
        bandwidth_hz=100e6,
        tx_power_dbm=23.0,
        antenna_height=10.0,
        noise_density_dbm_hz=-174.0,
        los_type='nlos'  # Non-line of sight
    )
    
    wireless_5g_los = Wireless5G(wireless_params_los)
    wireless_5g_nlos = Wireless5G(wireless_params_nlos)
    
    # Configuração da simulação
    config_los = SimulationConfig(
        # wireless=wireless_5g_los,
        tx_power_dbm=23.0
    )
    
    config_nlos = SimulationConfig(
        # wireless=wireless_5g_nlos,
        tx_power_dbm=23.0
    )
    
    # Calculadora de energia
    energy_calculator_los = EnergyCalculator(config_los)
    energy_calculator_nlos = EnergyCalculator(config_nlos)
    
    return energy_calculator_los, energy_calculator_nlos

# Funções para análise e geração de gráficos
def analyze_path_loss(wireless_5g_los, wireless_5g_nlos):
    """Analisa a perda de caminho (path loss) em função da distância para LOS e NLOS"""
    distances = np.linspace(10, 1000, 100)  # 10m a 1000m
    path_loss_los = [wireless_5g_los.calculate_path_loss(d) for d in distances]
    path_loss_nlos = [wireless_5g_nlos.calculate_path_loss(d) for d in distances]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, path_loss_los, 'b-', label='LOS')
    plt.plot(distances, path_loss_nlos, 'r-', label='NLOS')
    plt.xlabel('Distância (m)')
    plt.ylabel('Perda de Caminho (dB)')
    plt.title('Perda de Caminho vs. Distância')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def analyze_snr(wireless_5g_los, wireless_5g_nlos):
    """Analisa o SNR em função da distância para LOS e NLOS"""
    distances = np.linspace(10, 1000, 100)
    
    # Calculando SNR várias vezes para capturar a natureza estocástica
    num_samples = 30
    snr_los_samples = np.zeros((len(distances), num_samples))
    snr_nlos_samples = np.zeros((len(distances), num_samples))
    
    for i in range(num_samples):
        for j, distance in enumerate(distances):
            snr_los_samples[j, i] = 10 * np.log10(wireless_5g_los.calculate_snr(distance))
            snr_nlos_samples[j, i] = 10 * np.log10(wireless_5g_nlos.calculate_snr(distance))
    
    # Calculando média e desvio padrão
    snr_los_mean = np.mean(snr_los_samples, axis=1)
    snr_los_std = np.std(snr_los_samples, axis=1)
    snr_nlos_mean = np.mean(snr_nlos_samples, axis=1)
    snr_nlos_std = np.std(snr_nlos_samples, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, snr_los_mean, 'b-', label='LOS (média)')
    plt.fill_between(distances, snr_los_mean - snr_los_std, snr_los_mean + snr_los_std, 
                     color='b', alpha=0.2, label='LOS (desvio padrão)')
    plt.plot(distances, snr_nlos_mean, 'r-', label='NLOS (média)')
    plt.fill_between(distances, snr_nlos_mean - snr_nlos_std, snr_nlos_mean + snr_nlos_std, 
                     color='r', alpha=0.2, label='NLOS (desvio padrão)')
    plt.xlabel('Distância (m)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs. Distância')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def analyze_throughput(wireless_5g_los, wireless_5g_nlos):
    """Analisa a taxa de transferência em função da distância para LOS e NLOS"""
    distances = np.linspace(10, 1000, 100)
    
    # Calculando throughput várias vezes para capturar a natureza estocástica
    num_samples = 30
    throughput_los_samples = np.zeros((len(distances), num_samples))
    throughput_nlos_samples = np.zeros((len(distances), num_samples))
    
    for i in range(num_samples):
        for j, distance in enumerate(distances):
            throughput_los_samples[j, i] = wireless_5g_los.calculate_throughput(distance) / 1e6  # Mbps
            throughput_nlos_samples[j, i] = wireless_5g_nlos.calculate_throughput(distance) / 1e6  # Mbps
    
    # Calculando média e desvio padrão
    throughput_los_mean = np.mean(throughput_los_samples, axis=1)
    throughput_los_std = np.std(throughput_los_samples, axis=1)
    throughput_nlos_mean = np.mean(throughput_nlos_samples, axis=1)
    throughput_nlos_std = np.std(throughput_nlos_samples, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, throughput_los_mean, 'b-', label='LOS (média)')
    plt.fill_between(distances, throughput_los_mean - throughput_los_std, throughput_los_mean + throughput_los_std, 
                     color='b', alpha=0.2, label='LOS (desvio padrão)')
    plt.plot(distances, throughput_nlos_mean, 'r-', label='NLOS (média)')
    plt.fill_between(distances, throughput_nlos_mean - throughput_nlos_std, throughput_nlos_mean + throughput_nlos_std, 
                     color='r', alpha=0.2, label='NLOS (desvio padrão)')
    plt.xlabel('Distância (m)')
    plt.ylabel('Taxa de Transferência (Mbps)')
    plt.title('Taxa de Transferência vs. Distância')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def analyze_transmission_time(wireless_5g_los, wireless_5g_nlos):
    """Analisa o tempo de transmissão em função da distância e tamanho dos dados"""
    distances = np.linspace(10, 500, 20)
    data_sizes = np.linspace(0.1, 10, 20)  # MB
    
    # Criando grade de dados
    X, Y = np.meshgrid(distances, data_sizes)
    Z_los = np.zeros_like(X)
    Z_nlos = np.zeros_like(X)
    
    for i in range(len(distances)):
        for j in range(len(data_sizes)):
            Z_los[j, i] = wireless_5g_los.calculate_transmission_time(distances[i], data_sizes[j])
            Z_nlos[j, i] = wireless_5g_nlos.calculate_transmission_time(distances[i], data_sizes[j])
    
    # Limitando os valores para melhor visualização
    Z_los = np.clip(Z_los, 0, 10)
    Z_nlos = np.clip(Z_nlos, 0, 10)
    
    fig = plt.figure(figsize=(18, 7))
    
    # Gráfico para LOS
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_los, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Tamanho dos Dados (MB)')
    ax1.set_zlabel('Tempo de Transmissão (s)')
    ax1.set_title('Tempo de Transmissão (LOS)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Gráfico para NLOS
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_nlos, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax2.set_xlabel('Distância (m)')
    ax2.set_ylabel('Tamanho dos Dados (MB)')
    ax2.set_zlabel('Tempo de Transmissão (s)')
    ax2.set_title('Tempo de Transmissão (NLOS)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return plt.gcf()

def analyze_transmission_energy(energy_calc_los, energy_calc_nlos):
    """Analisa a energia de transmissão em função da distância e tamanho dos dados"""
    distances = np.linspace(10, 500, 20)
    data_sizes = np.linspace(0.1, 10, 20)  # MB
    
    # Criando grade de dados
    X, Y = np.meshgrid(distances, data_sizes)
    Z_los_static = np.zeros_like(X)
    Z_nlos_static = np.zeros_like(X)
    Z_los_dynamic = np.zeros_like(X)
    Z_nlos_dynamic = np.zeros_like(X)
    
    for i in range(len(distances)):
        for j in range(len(data_sizes)):
            Z_los_static[j, i] = energy_calc_los.calculate_transmission_energy(distances[i], data_sizes[j])
            Z_nlos_static[j, i] = energy_calc_nlos.calculate_transmission_energy(distances[i], data_sizes[j])
            Z_los_dynamic[j, i] = energy_calc_los.calculate_transmission_energy_dynamic(distances[i], data_sizes[j])
            Z_nlos_dynamic[j, i] = energy_calc_nlos.calculate_transmission_energy_dynamic(distances[i], data_sizes[j])
    
    # Limitando os valores para melhor visualização
    max_value = 100
    Z_los_static = np.clip(Z_los_static, 0, max_value)
    Z_nlos_static = np.clip(Z_nlos_static, 0, max_value)
    Z_los_dynamic = np.clip(Z_los_dynamic, 0, max_value)
    Z_nlos_dynamic = np.clip(Z_nlos_dynamic, 0, max_value)
    
    fig = plt.figure(figsize=(18, 14))
    
    # Gráfico para LOS estático
    ax1 = fig.add_subplot(221, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_los_static, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Tamanho dos Dados (MB)')
    ax1.set_zlabel('Energia (J)')
    ax1.set_title('Energia de Transmissão Estática (LOS)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Gráfico para NLOS estático
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_nlos_static, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_xlabel('Distância (m)')
    ax2.set_ylabel('Tamanho dos Dados (MB)')
    ax2.set_zlabel('Energia (J)')
    ax2.set_title('Energia de Transmissão Estática (NLOS)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # Gráfico para LOS dinâmico
    ax3 = fig.add_subplot(223, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_los_dynamic, cmap=cm.plasma, linewidth=0, antialiased=True)
    ax3.set_xlabel('Distância (m)')
    ax3.set_ylabel('Tamanho dos Dados (MB)')
    ax3.set_zlabel('Energia (J)')
    ax3.set_title('Energia de Transmissão Dinâmica (LOS)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # Gráfico para NLOS dinâmico
    ax4 = fig.add_subplot(224, projection='3d')
    surf4 = ax4.plot_surface(X, Y, Z_nlos_dynamic, cmap=cm.plasma, linewidth=0, antialiased=True)
    ax4.set_xlabel('Distância (m)')
    ax4.set_ylabel('Tamanho dos Dados (MB)')
    ax4.set_zlabel('Energia (J)')
    ax4.set_title('Energia de Transmissão Dinâmica (NLOS)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return plt.gcf()

def compare_computation_energy_models(energy_calc_los):
    """Compara os modelos estático e estocástico para energia computacional"""
    cpu_cycles_range = np.linspace(1e6, 1e9, 100)  # 1M a 1B ciclos
    frequency_ghz = 2.4
    kappa_nj = 1.5
    
    # Energia computacional estática
    static_energy = [energy_calc_los.calculate_computation_energy(cycles, kappa_nj) for cycles in cpu_cycles_range]
    
    # Energia computacional estocástica (calculada várias vezes)
    num_samples = 30
    stochastic_energy_samples = np.zeros((len(cpu_cycles_range), num_samples))
    
    for i in range(num_samples):
        for j, cycles in enumerate(cpu_cycles_range):
            stochastic_energy_samples[j, i] = energy_calc_los.calculate_computation_energy_stochastic(
                cycles, frequency_ghz, kappa_nj)
    
    stochastic_energy_mean = np.mean(stochastic_energy_samples, axis=1)
    stochastic_energy_std = np.std(stochastic_energy_samples, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_cycles_range / 1e6, static_energy, 'b-', label='Modelo Estático')
    plt.plot(cpu_cycles_range / 1e6, stochastic_energy_mean, 'r-', label='Modelo Estocástico (média)')
    plt.fill_between(cpu_cycles_range / 1e6, 
                     stochastic_energy_mean - stochastic_energy_std, 
                     stochastic_energy_mean + stochastic_energy_std, 
                     color='r', alpha=0.2, label='Modelo Estocástico (desvio padrão)')
    plt.xlabel('Ciclos de CPU (milhões)')
    plt.ylabel('Energia (J)')
    plt.title('Comparação dos Modelos de Energia Computacional')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def analyze_total_energy_scenario(energy_calc_los, energy_calc_nlos):
    """Analisa o consumo total de energia para diferentes cenários"""
    # Definindo diferentes distâncias
    distances = np.array([10, 50, 100, 200, 500, 1000])
    
    # Definindo 2 tipos de dispositivos
    device_source = Device(
        position=np.array([0, 0, 0]),
        frequency_ghz=2.4,
        kappa_nj=1.5
    )
    
    # Definindo 3 tipos de tarefas
    task_small = Task(cpu_cycles=1e7, data_size=0.5)  # Tarefa pequena
    task_medium = Task(cpu_cycles=5e7, data_size=2.0)  # Tarefa média
    task_large = Task(cpu_cycles=2e8, data_size=8.0)   # Tarefa grande
    
    tasks = [task_small, task_medium, task_large]
    task_names = ['Pequena', 'Média', 'Grande']
    
    # Preparando arrays para armazenar resultados
    results = []
    
    for d in distances:
        device_destination = Device(
            position=np.array([d, 0, 0]),
            frequency_ghz=2.0,
            kappa_nj=1.2
        )
        
        for los_type, energy_calc in [('LOS', energy_calc_los), ('NLOS', energy_calc_nlos)]:
            for task_idx, task in enumerate(tasks):
                # Calculando componentes de energia
                tx_energy_up = energy_calc.calculate_transmission_energy_dynamic(d, task.data_size, 'uplink')
                tx_energy_down = energy_calc.calculate_transmission_energy_dynamic(d, task.data_size, 'downlink')
                comp_energy = energy_calc.calculate_computation_energy_stochastic(
                    task.cpu_cycles * task.data_size, 
                    device_destination.frequency_ghz,
                    device_destination.kappa_nj
                )
                total_energy = tx_energy_up + tx_energy_down + comp_energy
                
                # Armazenando resultados
                results.append({
                    'Distância': d,
                    'Tipo': los_type,
                    'Tarefa': task_names[task_idx],
                    'Energia de Upload (J)': tx_energy_up,
                    'Energia de Download (J)': tx_energy_down,
                    'Energia Computacional (J)': comp_energy,
                    'Energia Total (J)': total_energy
                })
    
    # Convertendo para DataFrame para facilitar a análise
    df = pd.DataFrame(results)
    
    # Calculando as médias por tipo de tarefa, tipo de conexão e distância
    grouped_by_task_los_distance = df.groupby(['Tarefa', 'Tipo', 'Distância']).mean().reset_index()
    
    # Criando gráficos de barras empilhadas para análise da energia
    plt.figure(figsize=(20, 10))
    
    # Filtrando para diferentes distâncias
    distances_to_plot = [50, 200, 500]
    
    for i, d in enumerate(distances_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Filtrando dados para a distância específica
        df_filtered = grouped_by_task_los_distance[grouped_by_task_los_distance['Distância'] == d]
        
        # Preparando dados para o gráfico
        tasks = []
        upload_energy = []
        download_energy = []
        comp_energy = []
        
        for _, row in df_filtered.iterrows():
            tasks.append(f"{row['Tarefa']}-{row['Tipo']}")
            upload_energy.append(row['Energia de Upload (J)'])
            download_energy.append(row['Energia de Download (J)'])
            comp_energy.append(row['Energia Computacional (J)'])
        
        # Criando gráfico de barras empilhadas
        width = 0.7
        plt.bar(tasks, upload_energy, width, label='Upload')
        plt.bar(tasks, download_energy, width, bottom=upload_energy, label='Download')
        
        # Somando para calcular a base para computação
        upload_download = [u + d for u, d in zip(upload_energy, download_energy)]
        plt.bar(tasks, comp_energy, width, bottom=upload_download, label='Computação')
        
        plt.xlabel('Configuração')
        plt.ylabel('Energia (J)')
        plt.title(f'Consumo de Energia a {d}m')
        plt.xticks(rotation=45)
        plt.legend()
    
    plt.tight_layout()
    
    return plt.gcf()

def analyze_energy_efficiency(energy_calc_los, energy_calc_nlos):
    """Analisa a eficiência energética por bit transferido em função da distância"""
    distances = np.linspace(10, 500, 50)
    data_size_mb = 1.0  # 1 MB fixo
    
    # Calculando energia e eficiência
    energy_los = [energy_calc_los.calculate_transmission_energy(d, data_size_mb) for d in distances]
    energy_nlos = [energy_calc_nlos.calculate_transmission_energy(d, data_size_mb) for d in distances]
    
    # Convertendo MB para bits
    data_size_bits = data_size_mb * 8e6
    
    # Calculando eficiência (Joules por bit)
    efficiency_los = [e / data_size_bits for e in energy_los]
    efficiency_nlos = [e / data_size_bits for e in energy_nlos]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, efficiency_los, 'b-', label='LOS')
    plt.plot(distances, efficiency_nlos, 'r-', label='NLOS')
    plt.xlabel('Distância (m)')
    plt.ylabel('Energia por Bit (J/bit)')
    plt.title('Eficiência Energética vs. Distância')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def analyze_bandwidth_effect(energy_calc_los):
    """Analisa o efeito da largura de banda no consumo de energia"""
    original_bandwidth = energy_calc_los.config.wireless.params.bandwidth_hz
    
    bandwidths = np.linspace(20e6, 200e6, 5)  # 20MHz a 200MHz
    distances = np.linspace(10, 500, 20)
    data_size_mb = 5.0  # 5 MB
    
    energy_values = np.zeros((len(bandwidths), len(distances)))
    
    for i, bw in enumerate(bandwidths):
        # Alterando a largura de banda temporariamente
        energy_calc_los.config.wireless.params.bandwidth_hz = bw
        
        for j, d in enumerate(distances):
            energy_values[i, j] = energy_calc_los.calculate_transmission_energy(d, data_size_mb)
    
    # Restaurando o valor original
    energy_calc_los.config.wireless.params.bandwidth_hz = original_bandwidth
    
    plt.figure(figsize=(10, 6))
    for i, bw in enumerate(bandwidths):
        plt.plot(distances, energy_values[i], label=f'Largura de Banda: {bw/1e6:.0f} MHz')
    
    plt.xlabel('Distância (m)')
    plt.ylabel('Energia de Transmissão (J)')
    plt.title(f'Efeito da Largura de Banda no Consumo de Energia (Dados: {data_size_mb} MB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def analyze_tx_power_effect(energy_calc_los):
    """Analisa o efeito da potência de transmissão no consumo de energia"""
    original_tx_power = energy_calc_los.config.tx_power_dbm
    
    tx_powers = np.linspace(10, 30, 5)  # 10 dBm a 30 dBm
    distances = np.linspace(10, 500, 20)
    data_size_mb = 5.0  # 5 MB
    
    energy_values = np.zeros((len(tx_powers), len(distances)))
    throughput_values = np.zeros((len(tx_powers), len(distances)))
    
    for i, tx_power in enumerate(tx_powers):
        # Alterando a potência de transmissão temporariamente
        energy_calc_los.config.tx_power_dbm = tx_power
        energy_calc_los.config.wireless.params.tx_power_dbm = tx_power
        
        for j, d in enumerate(distances):
            energy_values[i, j] = energy_calc_los.calculate_transmission_energy(d, data_size_mb)
            throughput_values[i, j] = energy_calc_los.config.wireless.calculate_throughput(d) / 1e6  # Mbps
    
    # Restaurando o valor original
    energy_calc_los.config.tx_power_dbm = original_tx_power
    energy_calc_los.config.wireless.params.tx_power_dbm = original_tx_power
    
    # Figura 1: Energia vs Distância para diferentes potências de transmissão
    plt.figure(figsize=(18, 7))
    
    plt.subplot(1, 2, 1)
    for i, tx_power in enumerate(tx_powers):
        plt.plot(distances, energy_values[i], label=f'Potência Tx: {tx_power:.0f} dBm')
    
    plt.xlabel('Distância (m)')
    plt.ylabel('Energia de Transmissão (J)')
    plt.title(f'Efeito da Potência de Tx no Consumo de Energia (Dados: {data_size_mb} MB)')
    plt.grid(True)
    plt.legend()
    
    # Figura 2: Throughput vs Distância para diferentes potências de transmissão
    plt.subplot(1, 2, 2)
    for i, tx_power in enumerate(tx_powers):
        plt.plot(distances, throughput_values[i], label=f'Potência Tx: {tx_power:.0f} dBm')
    
    plt.xlabel('Distância (m)')
    plt.ylabel('Taxa de Transferência (Mbps)')
    plt.title('Efeito da Potência de Tx na Taxa de Transferência')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    return plt.gcf()

def analyze_dynamic_factors(energy_calc_los):
    """Analisa os fatores dinâmicos que afetam o consumo de energia"""
    # Distância fixa
    distance = 100.0  # metros
    data_size_mb = 5.0  # MB
    
    # Número de amostras
    num_samples = 1000
    
    # Parâmetros para análise de dispersão
    packet_error_rates = np.random.uniform(0.01, 0.05, num_samples)
    power_amplifier_efficiencies = np.random.normal(0.38, 0.02, num_samples)
    adaptive_tx_powers = np.random.uniform(-1, 1, num_samples)
    
    # Array para armazenar energia
    energy_values = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Armazenando o estado atual do gerador de números aleatórios
        state = np.random.get_state()
        
        # Definindo uma seed específica para esta amostra para tornar a função determinística
        np.random.seed(int(distance * 100) + hash('uplink') % 1000)
        
        # Forçando os valores específicos para esta amostra
        # Substitui np.random.uniform(0.01, 0.05) na função calculate_transmission_energy_dynamic
        # Este é um "hack" para controlar o comportamento da função sem modificar seu código
        np.random.uniform(0.01, 0.05)  # Consumindo uma chamada aleatória (packet_error_rate)
        np.random.geometric(1 - packet_error_rates[i])  # Consumindo outra chamada (num_retransmissions)
        np.random.uniform(-1, 1)  # Consumindo outra chamada (adaptive_tx_power)
        
        # Calculando energia
        energy_values[i] = energy_calc_los.calculate_transmission_energy_dynamic(distance, data_size_mb)
        
        # Restaurando o estado do gerador
        np.random.set_state(state)
    
    # Criando DataFrame para análise
    df = pd.DataFrame({
        'PER': packet_error_rates,
        'PA_Efficiency': power_amplifier_efficiencies,
        'TX_Power_Variation': adaptive_tx_powers,
        'Energy': energy_values
    })
    
    # Criando gráficos de dispersão
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Taxa de erro de pacote vs Energia
    axes[0].scatter(df['PER'], df['Energy'], alpha=0.5)
    axes[0].set_xlabel('Taxa de Erro de Pacote')
    axes[0].set_ylabel('Energia (J)')
    axes[0].set_title('Efeito da Taxa de Erro de Pacote')
    axes[0].grid(True)
    
    # Eficiência do amplificador vs Energia
    axes[1].scatter(df['PA_Efficiency'], df['Energy'], alpha=0.5)
    axes[1].set_xlabel('Eficiência do Amplificador de Potência')
    axes[1].set_ylabel('Energia (J)')
    axes[1].set_title('Efeito da Eficiência do Amplificador')
    axes[1].grid(True)
    
    # Variação da potência de Tx vs Energia
    axes[2].scatter(df['TX_Power_Variation'], df['Energy'], alpha=0.5)
    axes[2].set_xlabel('Variação da Potência de Tx (dB)')
    axes[2].set_ylabel('Energia (J)')
    axes[2].set_title('Efeito da Variação da Potência de Tx')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    return plt.gcf()

def analyze_computation_vs_communication(energy_calc_los):
    """Analisa o equilíbrio entre energia de computação e comunicação"""
    distances = np.linspace(10, 500, 20)
    data_sizes = np.linspace(0.1, 10, 20)  # MB
    
    # Parâmetros de computação
    cpu_cycles_per_mb = 1e7  # Ciclos de CPU por MB de dados
    frequency_ghz = 2.0
    kappa_nj = 1.5
    
    # Criando grade de dados
    X, Y = np.meshgrid(distances, data_sizes)
    Z_comm = np.zeros_like(X)  # Energia de comunicação
    Z_comp = np.zeros_like(X)  # Energia de computação
    Z_ratio = np.zeros_like(X)  # Razão entre energia de comunicação e computação
    
    for i in range(len(distances)):
        for j in range(len(data_sizes)):
            # Energia de comunicação (uplink + downlink)
            comm_energy = (
                energy_calc_los.calculate_transmission_energy(distances[i], data_sizes[j]) +
                energy_calc_los.calculate_transmission_energy(distances[i], data_sizes[j] * 0.1)  # Resposta geralmente menor
            )
            
            # Energia de computação
            comp_energy = energy_calc_los.calculate_computation_energy(
                int(cpu_cycles_per_mb * data_sizes[j]), kappa_nj)
            
            Z_comm[j, i] = comm_energy
            Z_comp[j, i] = comp_energy
            Z_ratio[j, i] = comm_energy / (comp_energy if comp_energy > 0 else 1e-10)
    
    # Limitando os valores para melhor visualização
    Z_ratio = np.clip(Z_ratio, 0, 10)
    
    # Criando gráfico de curvas de nível para a razão entre energia de comunicação e computação
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z_ratio, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Razão Comunicação/Computação')
    
    # Adicionando contornos para destacar regiões de interesse
    contours = plt.contour(X, Y, Z_ratio, levels=[0.5, 1.0, 2.0, 5.0], colors='white')
    plt.clabel(contours, inline=True, fontsize=8)
    
    plt.xlabel('Distância (m)')
    plt.ylabel('Tamanho dos Dados (MB)')
    plt.title('Razão entre Energia de Comunicação e Computação')
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

# Função principal para executar todas as análises e exibir os resultados
def main():
    # Configurando estilo dos gráficos
    plt.style.use('fivethirtyeight')
    sns.set_palette("colorblind")
    
    # Criando a configuração de simulação
    energy_calc_los, energy_calc_nlos = create_simulation_setup()
    
    # Análise 1: Perda de caminho (path loss)
    path_loss_fig = analyze_path_loss(energy_calc_los.config.wireless, energy_calc_nlos.config.wireless)
    path_loss_fig.savefig('path_loss_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 2: SNR
    snr_fig = analyze_snr(energy_calc_los.config.wireless, energy_calc_nlos.config.wireless)
    snr_fig.savefig('snr_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 3: Taxa de transferência (throughput)
    throughput_fig = analyze_throughput(energy_calc_los.config.wireless, energy_calc_nlos.config.wireless)
    throughput_fig.savefig('throughput_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 4: Tempo de transmissão
    transmission_time_fig = analyze_transmission_time(energy_calc_los.config.wireless, energy_calc_nlos.config.wireless)
    transmission_time_fig.savefig('transmission_time_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 5: Energia de transmissão
    transmission_energy_fig = analyze_transmission_energy(energy_calc_los, energy_calc_nlos)
    transmission_energy_fig.savefig('transmission_energy_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 6: Comparação dos modelos de energia computacional
    computation_energy_fig = compare_computation_energy_models(energy_calc_los)
    computation_energy_fig.savefig('computation_energy_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 7: Cenário completo de energia
    total_energy_fig = analyze_total_energy_scenario(energy_calc_los, energy_calc_nlos)
    total_energy_fig.savefig('total_energy_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 8: Eficiência energética
    energy_efficiency_fig = analyze_energy_efficiency(energy_calc_los, energy_calc_nlos)
    energy_efficiency_fig.savefig('energy_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 9: Efeito da largura de banda
    bandwidth_effect_fig = analyze_bandwidth_effect(energy_calc_los)
    bandwidth_effect_fig.savefig('bandwidth_effect_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 10: Efeito da potência de transmissão
    tx_power_effect_fig = analyze_tx_power_effect(energy_calc_los)
    tx_power_effect_fig.savefig('tx_power_effect_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 11: Fatores dinâmicos
    dynamic_factors_fig = analyze_dynamic_factors(energy_calc_los)
    dynamic_factors_fig.savefig('dynamic_factors_analysis.png', dpi=300, bbox_inches='tight')
    
    # Análise 12: Equilíbrio entre computação e comunicação
    comp_vs_comm_fig = analyze_computation_vs_communication(energy_calc_los)
    comp_vs_comm_fig.savefig('comp_vs_comm_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Análise concluída. Todas as figuras foram salvas.")

if __name__ == "__main__":
    main()