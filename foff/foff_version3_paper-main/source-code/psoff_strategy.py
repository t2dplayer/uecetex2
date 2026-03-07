import numpy as np
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
from typing import List, Tuple
import logging

class Particle:
    """
    Representa uma partícula no algoritmo PSO para offloading de tarefas.
    Cada partícula contém uma possível solução (alocação de dispositivos para tarefas).
    """
    def __init__(self, num_tasks: int, num_devices: int):
        self.position = np.random.rand(num_tasks, num_devices)  # Matriz de posição (probabilidades)
        self.velocity = np.zeros((num_tasks, num_devices))      # Matriz de velocidade
        self.best_position = self.position.copy()               # Melhor posição individual
        self.fitness = float('inf')                             # Valor de fitness atual
        self.best_fitness = float('inf')                        # Melhor valor de fitness

    def update_position(self):
        """Atualiza a posição da partícula baseada na velocidade atual"""
        self.position += self.velocity
        
        # Restringe os valores entre 0 e 1
        self.position = np.clip(self.position, 0, 1)
        
        # Normaliza as linhas para que a soma seja 1 (restrição do problema)
        row_sums = self.position.sum(axis=1, keepdims=True)
        self.position = self.position / row_sums
        
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        Atualiza a velocidade da partícula
        
        Args:
            global_best_position: Melhor posição encontrada por qualquer partícula
            w: Fator de inércia
            c1: Coeficiente cognitivo (influência da melhor posição individual)
            c2: Coeficiente social (influência da melhor posição global)
        """
        r1 = np.random.rand(self.position.shape[0], self.position.shape[1])
        r2 = np.random.rand(self.position.shape[0], self.position.shape[1])
        
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component
        
        # Limita a velocidade para evitar explosão
        self.velocity = np.clip(self.velocity, -0.2, 0.2)

    def update_best_position(self):
        """Atualiza a melhor posição da partícula se a posição atual for melhor"""
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

class PSOFFStrategy(OffloadingStrategy):
    """
    Implementação da estratégia de offloading baseada em PSO (Particle Swarm Optimization).
    """
    def __init__(self, simulator, num_particles=20, max_iterations=30, w=0.7, c1=1.5, c2=1.5, gamma=0.85):
        """
        Inicializa a estratégia PSOFF.
        
        Args:
            simulator: O simulador de offloading
            num_particles: Número de partículas no enxame
            max_iterations: Número máximo de iterações
            w: Fator de inércia para atualização da velocidade
            c1: Coeficiente cognitivo
            c2: Coeficiente social
            gamma: Peso relativo do atraso vs. consumo de energia na função objetivo
        """
        super().__init__(simulator)
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma  # Peso para balancear atraso vs energia (como no artigo)
        self.logger = logging.getLogger(__name__)        
        
    def calculate_fitness(self, particle: Particle, tasks, devices) -> float:
        """
        Calcula o fitness de uma partícula (custo total = γ*atraso + (1-γ)*energia).
        
        Args:
            particle: A partícula a ser avaliada
            tasks: Lista de tarefas
            devices: Lista de dispositivos
            
        Returns:
            O valor de fitness (custo total) da partícula
        """
        total_latency = 0
        total_energy = 0
        
        # Para cada tarefa
        for i, task in enumerate(tasks):
            # A posição da partícula pode ser interpretada como probabilidades
            # Nós selecionamos o dispositivo com maior probabilidade
            device_idx = np.argmax(particle.position[i])
            device = devices[device_idx]
            
            # Calcula métricas de latência e energia
            latency, energy = self.simulator._calculate_task_metrics(task, device)
            
            total_latency += latency
            total_energy += energy
        
        # Calcula o custo total de acordo com a equação (23) do artigo
        total_cost = self.gamma * total_latency + (1 - self.gamma) * total_energy
        
        return total_cost
        
    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> Tuple[Device, None]:
        """
        Executa a estratégia PSOFF para encontrar a melhor alocação de dispositivos para a tarefa atual.
        
        Args:
            tasks: Lista de todas as tarefas
            current_task_index: Índice da tarefa atual
            device_manager: Gerenciador de dispositivos
            
        Returns:
            O dispositivo selecionado para a tarefa
        """
        current_task = tasks[current_task_index]
        devices = device_manager.devices
        
        # Se houver apenas um dispositivo, retorne-o
        if len(devices) <= 1:
            return devices[0], None
            
        # Se a decisão for apenas para uma tarefa, podemos simplificar o PSO
        # Vamos avaliar cada dispositivo e escolher o melhor
        best_device = None
        best_cost = float('inf')
        
        for device in devices:
            latency, energy = self.simulator._calculate_task_metrics(current_task, device)
            cost = self.gamma * latency + (1 - self.gamma) * energy
            
            if cost < best_cost:
                best_cost = cost
                best_device = device
        self.logger.debug(f"PSOFF: Task {current_task_index} scheduled to device {best_device.device_id}")
        return best_device, None
    
    def execute_batch(self, tasks, device_manager: DeviceManager) -> List[Device]:
        """
        Executa a estratégia PSOFF para um conjunto de tarefas, otimizando a alocação global.
        Este método não é chamado pelo simulador, mas pode ser integrado em uma versão futura.
        
        Args:
            tasks: Lista de todas as tarefas
            device_manager: Gerenciador de dispositivos
            
        Returns:
            Lista de dispositivos selecionados para cada tarefa
        """
        devices = device_manager.devices
        num_tasks = len(tasks)
        num_devices = len(devices)
        
        # Inicializa o enxame de partículas
        particles = [Particle(num_tasks, num_devices) for _ in range(self.num_particles)]
        global_best_position = np.zeros((num_tasks, num_devices))
        global_best_fitness = float('inf')
        
        # Executa o algoritmo PSO
        for _ in range(self.max_iterations):
            for particle in particles:
                # Calcula o fitness da partícula
                particle.fitness = self.calculate_fitness(particle, tasks, devices)
                
                # Atualiza a melhor posição da partícula
                particle.update_best_position()
                
                # Atualiza a melhor posição global
                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = particle.position.copy()
            
            # Atualiza a velocidade e posição de cada partícula
            for particle in particles:
                particle.update_velocity(global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
        
        # Converte a melhor solução encontrada em uma lista de dispositivos
        device_indices = np.argmax(global_best_position, axis=1)
        selected_devices = [devices[idx] for idx in device_indices]
        
        return selected_devices
    
    def __str__(self):
        """Retorna o nome da estratégia"""
        return 'PSOFF'
    
    def __repr__(self):
        """Representação em string da estratégia"""
        return self.__str__()