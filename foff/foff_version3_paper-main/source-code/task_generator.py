import csv
import random
import numpy as np

def generate_adversarial_tasks(num_tasks_per_line, num_lines, simulation_time, workload = 1, output_file="adversarial_tasks.csv"):
    """
    Gera um conjunto de tarefas adversariais para algoritmos gulosos.
    
    Args:
        num_tasks_per_line: Número de tarefas por linha no CSV
        num_lines: Número total de linhas no CSV
        simulation_time: Tempo total da simulação em segundos
        output_file: Nome do arquivo de saída
    """
    all_tasks = []
    
    for line in range(num_lines):
        line_tasks = []
        
        # Determina o tipo de padrão adversarial para esta linha
        pattern_type = line % 5  # Alterna entre 5 padrões adversariais diferentes
        
        # Sementes diferentes para cada linha para garantir variabilidade
        random.seed(1000 + line)
        np.random.seed(1000 + line)
        
        for i in range(num_tasks_per_line):
            # Calcula o ID da tarefa
            task_id = i
            
            # Define os parâmetros com base no padrão adversarial escolhido
            if pattern_type == 0:
                # Padrão 1: Primeiras tarefas pequenas com alta prioridade, depois grandes com baixa prioridade
                # Algoritmos gulosos podem escolher as pequenas primeiro, mas isso causa gargalos para as grandes
                if i < num_tasks_per_line // 3:
                    data_size = random.randint(10*workload, 50*workload)
                    cpu_cycles = random.randint(15*workload, 19*workload)
                    arrival_time = (i * simulation_time * 0.1) / (num_tasks_per_line // 3)
                elif i < 2 * (num_tasks_per_line // 3):
                    data_size = random.randint(300*workload, 600*workload)
                    cpu_cycles = random.randint(8*workload, 12*workload)
                    arrival_time = simulation_time * 0.1 + ((i - num_tasks_per_line // 3) * simulation_time * 0.2) / (num_tasks_per_line // 3)
                else:
                    data_size = random.randint(800*workload, 1020*workload)
                    cpu_cycles = random.randint(1*workload, 5*workload)
                    arrival_time = simulation_time * 0.3 + ((i - 2 * (num_tasks_per_line // 3)) * simulation_time * 0.1) / (num_tasks_per_line - 2 * (num_tasks_per_line // 3))
                    
            elif pattern_type == 1:
                # Padrão 2: Alternância de cargas para criar desequilíbrio de recursos
                if i % 3 == 0:
                    data_size = random.randint(800*workload, 1020*workload)
                    cpu_cycles = random.randint(1*workload, 3*workload)
                elif i % 3 == 1:
                    data_size = random.randint(50*workload, 200*workload)
                    cpu_cycles = random.randint(17*workload, 19*workload)
                else:
                    data_size = random.randint(400*workload, 600*workload)
                    cpu_cycles = random.randint(8*workload, 12*workload)
                arrival_time = (i * simulation_time) / num_tasks_per_line + random.uniform(0, simulation_time * 0.02)
                
            elif pattern_type == 2:
                # Padrão 3: Grupo de tarefas com chegada simultânea
                # Cria concorrência por recursos em momentos específicos
                group = i // (num_tasks_per_line // 4)
                data_size = random.randint(300*workload + group * 200, 500*workload + group * 200)
                cpu_cycles = random.randint(5*workload, 15*workload)
                arrival_time = (group * simulation_time * 0.25) + random.uniform(0, simulation_time * 0.04)
                
            elif pattern_type == 3:
                # Padrão 4: Tarefas similares, mas com tempos de chegada que favorecem
                # decisões míopes que são ruins a longo prazo
                data_size = random.randint(400*workload, 800*workload)
                cpu_cycles = random.randint(8*workload, 12*workload)
                
                # Cria grupos que chegam em ondas
                if i < num_tasks_per_line // 4:
                    arrival_time = (i * simulation_time * 0.2) / (num_tasks_per_line // 4)
                elif i < num_tasks_per_line // 2:
                    arrival_time = simulation_time * 0.2 + ((i - num_tasks_per_line // 4) * simulation_time * 0.3) / (num_tasks_per_line // 4)
                elif i < 3 * (num_tasks_per_line // 4):
                    arrival_time = simulation_time * 0.5 + ((i - num_tasks_per_line // 2) * simulation_time * 0.3) / (num_tasks_per_line // 4)
                else:
                    arrival_time = simulation_time * 0.8 + ((i - 3 * (num_tasks_per_line // 4)) * simulation_time * 0.2) / (num_tasks_per_line // 4)
                    
            else:  # pattern_type == 4
                # Padrão 5: Tarefas CPU-intensivas no início, depois tarefas de dados intensivos
                # Algoritmos gulosos podem priorizar incorretamente
                if i < num_tasks_per_line // 2:
                    data_size = random.randint(10*workload, 100*workload)
                    cpu_cycles = random.randint(15*workload, 19*workload)
                    arrival_time = (i * simulation_time * 0.4) / (num_tasks_per_line // 2)
                else:
                    data_size = random.randint(700*workload, 1020*workload)
                    cpu_cycles = random.randint(1*workload, 5*workload)
                    arrival_time = simulation_time * 0.4 + ((i - num_tasks_per_line // 2) * simulation_time * 0.6) / (num_tasks_per_line - num_tasks_per_line // 2)
            
            # Garante que o tempo de chegada não ultrapasse o tempo de simulação
            arrival_time = min(arrival_time, simulation_time * 0.95)
            
            # Formata a tarefa como string
            task_str = f"Task(data_size={data_size}, cpu_cycles={cpu_cycles}, task_id={task_id}, arrival_time={arrival_time})"
            line_tasks.append(task_str)
            
        all_tasks.append(line_tasks)
    
    # Salva as tarefas no arquivo CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for line in all_tasks:
            writer.writerow(line)
    
    print(f"Arquivo gerado com sucesso: {output_file}")
    print(f"Geradas {num_lines} linhas com {num_tasks_per_line} tarefas cada.")
    print(f"Tempo total de simulação: {simulation_time} segundos")

if __name__ == "__main__":
    print("Gerador de Tarefas Adversariais para Algoritmos Gulosos")
    print("------------------------------------------------------")
    print("Este script gera tarefas especialmente projetadas para desafiar algoritmos gulosos.")
    
    try:
        num_tasks = int(input("Quantas tarefas por linha? "))
        num_lines = int(input("Quantas linhas no arquivo? "))
        simulation_time = float(input("Tempo total da simulação (segundos)? "))
        workload = 5
        output_file = input("Nome do arquivo de saída (padrão: adversarial_tasks.csv): ")
        if not output_file:
            output_file = "adversarial_tasks.csv"
        if not output_file.endswith('.csv'):
            output_file += '.csv'
             
        generate_adversarial_tasks(num_tasks, num_lines, simulation_time, workload, output_file)
        
    except ValueError:
        print("Erro: Por favor, insira valores numéricos válidos.")