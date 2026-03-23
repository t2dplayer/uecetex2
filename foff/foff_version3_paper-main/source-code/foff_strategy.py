from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
from fuzzy_topsis import *
import logging        

criteria_types = [Criteria.COST, Criteria.COST]

# Versão de transformação de variável linguística em NFT
# (a, b, c)
# a: primeiro valor de pertinência
# b: maior valor de pertinência
# c: último valor de pertinência
def extract_nft_map(memberships, linguistic_variable):
    numeros_fuzzy = {}
    for i, term in enumerate(linguistic_variable.terms):
        # Encontra os índices onde a função de pertinência é maior que zero
        indices = np.where(linguistic_variable[term].mf > 0)[0]

        if not indices.size:  # se indices estiver vazio, pula o termo
            numeros_fuzzy[i] = (0, 0, 0)  # ou pode lançar um erro, se preferir
            continue

        min_index = indices[0]
        max_index = indices[-1]

        a = linguistic_variable.universe[min_index]
        b = linguistic_variable.universe[linguistic_variable[term].mf == linguistic_variable[term].mf.max()][0]
        c = linguistic_variable.universe[max_index]

        numeros_fuzzy[i] = (a, b, c)
    max_membership_index = np.argmax(memberships)
    return numeros_fuzzy[max_membership_index]

# Versão de transformação de variável linguística em NFT
# (a, b, c)
# a: média ponderada de a entre todos os termos
# b: média ponderada de b entre todos os termos
# c: média ponderada de c entre todos os termos
def extract_nft_weighted(memberships, linguistic_variable):
    weighted_sum_a = 0
    weighted_sum_b = 0
    weighted_sum_c = 0
    total_weight = 0
    for i, term in enumerate(linguistic_variable.terms):
        indices = np.where(linguistic_variable[term].mf > 0)[0]
        if not indices.size:
            continue
        min_index = indices[0]
        max_index = indices[-1]
        a = linguistic_variable.universe[min_index]
        b = linguistic_variable.universe[linguistic_variable[term].mf == linguistic_variable[term].mf.max()][0]
        c = linguistic_variable.universe[max_index]
        weight_val = memberships[i]
        weighted_sum_a += a * weight_val
        weighted_sum_b += b * weight_val
        weighted_sum_c += c * weight_val
        total_weight += weight_val
    if total_weight == 0:
        return 0, 0, 0
    return weighted_sum_a / total_weight, weighted_sum_b / total_weight, weighted_sum_c / total_weight

def extract_triangular_fuzzy_parameters(fuzzy_set, variable): #<-- adicionado parametro variable.
    """Extrai os parâmetros (mínimo, médio, máximo) de um conjunto fuzzy triangular."""
    indices = np.where(fuzzy_set.mf > 0)[0]
    min_index = indices[0]
    max_index = indices[-1]
    a = variable.universe[min_index] #<-- usando variable.universe
    b = variable.universe[fuzzy_set.mf == fuzzy_set.mf.max()][0] #<-- usando variable.universe
    c = variable.universe[max_index] #<-- usando variable.universe
    return np.array([a, b, c])

def create_weight_vector(importances, weight_variable):
    """Cria o vetor de pesos W_tilde com NFTs triangulares."""
    W_tilde = np.zeros((len(importances), 3))
    for i, importance_label in enumerate(importances):
        min_val, mid_val, max_val = extract_triangular_fuzzy_parameters(weight_variable[importance_label], weight_variable) # Passando weight_variable para a função
        W_tilde[i] = [min_val, mid_val, max_val]

    return W_tilde

class DFOFFStrategy(OffloadingStrategy):
    def __init__(self, simulator, omega: List[str] = ['vl', 'vh']):
        super().__init__(simulator)
        self.ratings = ["vp", "p", "mp", "f", "mg", "g", "vg"]
        self.importances = ["vl", "l", "ml", "m", "mh", "h", "vh"]
        self.fuzzy_resolution = self.simulator.config.fuzzy_resolution
        self.omega = omega
        self.logger = logging.getLogger(__name__)

    def set_omega(self, omega: List[str]):
        self.omega = omega

    def calculate(self, task, devices):        
        if len(devices) < 2:
            return np.zeros((1, len(devices), 2))
        result = np.zeros((1, len(devices), 2))
        for i, device in enumerate(devices):
            total_latency, total_energy = self.simulator._calculate_task_metrics(task, device)
            result[0, i] = [total_latency, total_energy]
        return result

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager):
        task = tasks[current_task_index]
        results = self.calculate(task, device_manager.devices)        
        latency = ctrl.Antecedent(np.linspace(results[:, :, 0].min(), results[:, :, 0].max(), self.fuzzy_resolution), 'latency')
        latency.automf(number=7, names=self.ratings, invert=True)  # Inverte para critério de custo
        energy = ctrl.Antecedent(np.linspace(results[:, :, 1].min(), results[:, :, 1].max(), self.fuzzy_resolution), 'energy')
        energy.automf(number=7, names=self.ratings, invert=True)  # Inverte para critério de custo
        weight = ctrl.Antecedent(universe=np.linspace(0, 1, self.fuzzy_resolution), label="weight")
        weight.automf(number=7, names=self.importances, invert=False)
        # omega = np.array(['vl', 'vh'])
        W_tilde = create_weight_vector(self.omega, weight)
        D_tilde = self.make_decision_matrix(
            np.array([task]), 
            device_manager.devices,
            latency, energy, results)
        Cci = fuzzy_topsis(D_tilde, W_tilde, criteria_types)
        server_names = [f"{device.device_id}" for device in device_manager.devices ]
        # Criar uma lista 2D de resultados
        gamma = 0.85
        total_cost = 0
        for r in results:
            for v in r:
                total_cost += gamma * v[0] + (1 - gamma) * v[1]
        for r in results:
            for i, v in enumerate(r):
                cost = gamma * v[0] + (1 - gamma) * v[1]
                # print(Cci[i])
                Cci[i] -= (cost/total_cost)
        results = [[server_names[i], Cci[i]] for i in range(len(server_names))]
        
        # Classificar as alternativas
        ranked_alternatives = sorted(results, key=lambda item: item[1], reverse=True)
        device_index = int(ranked_alternatives[0][0])
        self.logger.debug(f"FOFF: Task {current_task_index} scheduled to device {device_index}")
        return device_manager.get_device(device_index), (self.omega, ranked_alternatives)
        
    def __str__(self):
        return 'FOFF'
    
    def __repr__(self):
        return self.__str__()
    
    def make_decision_matrix(self, tasks, devices, latency, energy, tasks_performances):
        D_tilde = np.zeros(shape=(len(devices), 2, 3))
        for task_index in range(tasks.shape[0]):
            for device_index in range(len(devices)):
                current_latency = tasks_performances[task_index, device_index, 0]
                memberships = np.array([fuzz.interp_membership(latency.universe, latency[term].mf, current_latency) for term in self.ratings])
                a, b, c = extract_nft_map(memberships, latency)
                D_tilde[device_index][0] += np.array([a, b, c])
                current_energy_comsuption = tasks_performances[task_index, device_index, 1]
                memberships = np.array([fuzz.interp_membership(energy.universe, energy[term].mf, current_energy_comsuption) for term in self.ratings])
                a, b, c = extract_nft_map(memberships, energy)
                D_tilde[device_index][1] += np.array([a, b, c])
        return D_tilde
    
class WFOFFStrategy(OffloadingStrategy):
    def __init__(self, simulator, omega: List[str] = ['vl', 'vh']):
        super().__init__(simulator)
        self.ratings = ["vp", "p", "mp", "f", "mg", "g", "vg"]
        self.importances = ["vl", "l", "ml", "m", "mh", "h", "vh"]
        self.fuzzy_resolution = self.simulator.config.fuzzy_resolution
        self.omega = omega

    def set_omega(self, omega: List[str]):
        self.omega = omega

    def calculate(self, task, devices):        
        if len(devices) < 2:
            return np.zeros((1, len(devices), 2))
        result = np.zeros((1, len(devices), 2))
        for i, device in enumerate(devices):
            total_latency, total_energy = self.simulator._calculate_task_metrics(task, device)
            result[0, i] = [total_latency, total_energy]
        return result

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager):
        task = tasks[current_task_index]
        results = self.calculate(task, device_manager.devices)        
        latency = ctrl.Antecedent(np.linspace(results[:, :, 0].min(), results[:, :, 0].max(), self.fuzzy_resolution), 'latency')
        latency.automf(number=7, names=self.ratings, invert=True)  # Inverte para critério de custo
        energy = ctrl.Antecedent(np.linspace(results[:, :, 1].min(), results[:, :, 1].max(), self.fuzzy_resolution), 'energy')
        energy.automf(number=7, names=self.ratings, invert=True)  # Inverte para critério de custo
        weight = ctrl.Antecedent(universe=np.linspace(0, 1, self.fuzzy_resolution), label="weight")
        weight.automf(number=7, names=self.importances, invert=False)
        # omega = np.array(['vl', 'vh'])
        W_tilde = create_weight_vector(self.omega, weight)
        D_tilde = self.make_decision_matrix(
            np.array([task]), 
            device_manager.devices, 
            latency, energy, results)
        Cci = fuzzy_topsis(D_tilde, W_tilde, criteria_types)
        server_names = [f"{device.device_id}" for device in device_manager.devices ]
        # Criar uma lista 2D de resultados
        results = [[server_names[i], Cci[i]] for i in range(len(server_names))]
        # Classificar as alternativas
        ranked_alternatives = sorted(results, key=lambda item: item[1], reverse=True)
        device_index = int(ranked_alternatives[0][0])
        return device_manager.get_device(device_index), (self.omega, ranked_alternatives)
        
    def __str__(self):
        return 'WFOFF'
    
    def __repr__(self):
        return self.__str__()
    
    def make_decision_matrix(self, tasks, devices, latency, energy, tasks_performances):
        D_tilde = np.zeros(shape=(len(devices), 2, 3))
        for task_index in range(tasks.shape[0]):
            for device_index in range(len(devices)):
                current_latency = tasks_performances[task_index, device_index, 0]
                memberships = np.array([fuzz.interp_membership(latency.universe, latency[term].mf, current_latency) for term in self.ratings])
                a, b, c = extract_nft_weighted(memberships, latency)
                D_tilde[device_index][0] += np.array([a, b, c])
                current_energy_comsuption = tasks_performances[task_index, device_index, 1]
                memberships = np.array([fuzz.interp_membership(energy.universe, energy[term].mf, current_energy_comsuption) for term in self.ratings])
                a, b, c = extract_nft_weighted(memberships, energy)
                D_tilde[device_index][1] += np.array([a, b, c])
        return D_tilde


