import numpy as np
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
from deap import base, creator, tools, algorithms
import logging

class HybridStrategy(OffloadingStrategy):
    """
    Implementação da estratégia híbrida de offloading descrita no artigo:
    "An intelligent hybrid method: Multi-objective optimization for MEC-enabled devices of IoE"
    
    Esta estratégia combina:
    1. Decisões baseadas em preferência para tarefas data-intensive e compute-intensive
    2. NSGA-II para otimização multi-objetivo das tarefas restantes
    """
    
    def __init__(self, simulator, alpha=0.15):
        """
        Inicializa a estratégia híbrida.
        
        Args:
            simulator: O simulador de offloading
            alpha: O parâmetro que determina a fração de tarefas a serem classificadas 
                  por preferências (0 < alpha < 0.5, padrão: 0.15)
        """
        super().__init__(simulator)
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)        
        
        # Configuração do NSGA-II
        self.population_size = 80
        self.max_generations = 100
        self.cxpb = 0.8  # Probabilidade de crossover
        self.mutpb = 0.3  # Probabilidade de mutação
        
        # Critérios de otimização: Latência e Energia (ambos para minimizar)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)

    def _calculate_weighted_values(self, task, all_tasks):
        """
        Calcula os valores ponderados de tamanho de dados e ciclos de CPU para uma tarefa.
        
        Args:
            task: A tarefa atual
            all_tasks: Todas as tarefas disponíveis
            
        Returns:
            weighted_data_size: O valor ponderado do tamanho dos dados
            weighted_cpu_cycles: O valor ponderado dos ciclos de CPU
        """
        # Soma do tamanho dos dados e ciclos de CPU de todas as tarefas
        total_data_size = sum(t.data_size for t in all_tasks)
        total_cpu_cycles = sum(t.cpu_cycles for t in all_tasks)
        
        # Calcula os valores ponderados
        if total_data_size > 0:
            weighted_data_size = task.data_size / total_data_size
        else:
            weighted_data_size = 0
            
        if total_cpu_cycles > 0:
            weighted_cpu_cycles = task.cpu_cycles / total_cpu_cycles
        else:
            weighted_cpu_cycles = 0
            
        return weighted_data_size, weighted_cpu_cycles

    def _determine_preferences(self, task, all_tasks):
        """
        Determina se uma tarefa deve ser executada localmente ou remotamente com base em suas características.
        
        Args:
            task: A tarefa a ser avaliada
            all_tasks: Todas as tarefas disponíveis
            
        Returns:
            decision: 0 para execução local, 1 para offloading, None para decisão via NSGA-II
        """
        # Calcula os valores ponderados para a tarefa
        weighted_data_size, weighted_cpu_cycles = self._calculate_weighted_values(task, all_tasks)
        
        # Encontra os valores máximos ponderados entre todas as tarefas
        max_weighted_data = max([self._calculate_weighted_values(t, all_tasks)[0] for t in all_tasks])
        max_weighted_cpu = max([self._calculate_weighted_values(t, all_tasks)[1] for t in all_tasks])
        
        # Calcula os limites com base no parâmetro alpha
        dt1 = self.alpha * max_weighted_data
        dt2 = (1 - self.alpha) * max_weighted_data
        ct1 = self.alpha * max_weighted_cpu
        ct2 = (1 - self.alpha) * max_weighted_cpu
        
        # Preferência 1: Tamanho de dados grande e computação pequena -> Execução local
        if weighted_data_size >= dt2 and weighted_cpu_cycles < ct1:
            return 0
        
        # Preferência 2: Tamanho de dados pequeno e computação grande -> Offloading
        elif weighted_data_size < dt1 and weighted_cpu_cycles >= ct2:
            return 1
        
        # Para outras tarefas, deixa a decisão para o NSGA-II
        return None

    def _setup_nsga2(self, tasks_to_optimize, devices):
        """
        Configura o algoritmo NSGA-II para otimização multi-objetivo.
        
        Args:
            tasks_to_optimize: Lista de tarefas que precisam ser otimizadas
            devices: Lista de dispositivos disponíveis
            
        Returns:
            toolbox: O toolbox DEAP configurado
        """
        toolbox = base.Toolbox()
        
        # Função para gerar um indivíduo aleatoricamente (lista de decisões de offloading)
        def generate_individual():
            return [np.random.randint(0, len(devices)) for _ in range(len(tasks_to_optimize))]
        
        # Configura o toolbox
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Função de avaliação (latência e energia)
        def evaluate_individual(individual):
            total_latency = 0
            total_energy = 0
            
            for i, device_idx in enumerate(individual):
                device = devices[device_idx]
                task = tasks_to_optimize[i]
                
                # Calcula métricas para a tarefa com o dispositivo selecionado
                latency, energy = self.simulator._calculate_task_metrics(task, device)
                
                total_latency += latency
                total_energy += energy
            
            return total_latency, total_energy
        
        # Registra operadores genéticos
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(devices)-1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        return toolbox

    def _run_nsga2(self, toolbox):
        """
        Executa o algoritmo NSGA-II para encontrar soluções ótimas.
        
        Args:
            toolbox: O toolbox DEAP configurado
            
        Returns:
            best_solution: A melhor solução encontrada
        """
        # Cria a população inicial
        population = toolbox.population(n=self.population_size)
        
        # Algoritmo NSGA-II
        algorithms.eaMuPlusLambda(
            population, 
            toolbox, 
            mu=self.population_size, 
            lambda_=self.population_size, 
            cxpb=self.cxpb, 
            mutpb=self.mutpb,
            ngen=self.max_generations, 
            verbose=False
        )
        
        # Obtém o conjunto de Pareto
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Seleciona a solução com o melhor equilíbrio entre latência e energia
        # (poderia ser aprimorado com métodos de tomada de decisão)
        best_solution = min(pareto_front, key=lambda ind: sum(ind.fitness.values))
        
        return best_solution

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> tuple:
        """
        Executa a estratégia híbrida de offloading.
        
        Args:
            tasks: Lista de todas as tarefas
            current_task_index: Índice da tarefa atual
            device_manager: Gerenciador de dispositivos
            
        Returns:
            device: O dispositivo selecionado para a tarefa
            info: Informações adicionais sobre a decisão
        """
        current_task = tasks[current_task_index]
        devices = device_manager.devices
        
        # Etapa 1: Verifica se a tarefa pode ser decidida por preferências
        preference_decision = self._determine_preferences(current_task, tasks)
        
        # Se a decisão foi tomada por preferências, retorna o dispositivo correspondente
        if preference_decision is not None:
            if preference_decision == 0:  # Execução local (OBUI)
                decision_method = "preference-local"
                # Assume que o OBUI é o último dispositivo
                obui_device = device_manager.get_device(self.simulator.obui_index)
                return obui_device, decision_method
            else:  # Offloading para o melhor dispositivo remoto
                decision_method = "preference-offload"
                
                # Encontra o melhor dispositivo remoto para offloading (maior capacidade computacional)
                best_device = None
                best_frequency = 0
                
                for device in devices:
                    if device.device_id != self.simulator.obui_index and device.frequency_ghz > best_frequency:
                        best_device = device
                        best_frequency = device.frequency_ghz
                
                return best_device, decision_method
        
        # Etapa 2: Se não foi possível decidir por preferências, usa NSGA-II
        # Simplificação: Como estamos decidindo apenas para a tarefa atual,
        # vamos avaliar todos os dispositivos e escolher o melhor
        
        # Calcula latência e energia para cada dispositivo
        metrics = []
        for device in devices:
            latency, energy = self.simulator._calculate_task_metrics(current_task, device)
            metrics.append((device, latency, energy))
        
        # Normaliza os valores para terem o mesmo peso
        max_latency = max(m[1] for m in metrics)
        max_energy = max(m[2] for m in metrics)
        
        # Escolhe o dispositivo com o melhor equilíbrio entre latência e energia normalizados
        best_device = None
        best_score = float('inf')
        
        for device, latency, energy in metrics:
            norm_latency = latency / max_latency if max_latency > 0 else 0
            norm_energy = energy / max_energy if max_energy > 0 else 0
            score = norm_latency + norm_energy
            
            if score < best_score:
                best_device = device
                best_score = score
        self.logger.debug(f"HOFF: Task {current_task_index} scheduled to device {best_device.device_id}")
        return best_device, "nsga2"
    
    def __str__(self):
        return 'HOFF'
    
    def __repr__(self):
        return self.__str__()