import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
from scipy.special import psi, betaln

# =================================================================================
# 1. CONFIGURAÇÕES ("THE GOLDILOCKS ZONE")
# =================================================================================
N_ARMS = 3
N_PLAYERS = 10
N_STEPS = 3000
MAX_QUEUE = 100.0

# O Ponto de Equilíbrio Matemático:
# Capacidade Máxima do Sistema (com 10 canais e split perfeito 5/5) ~= 4.5 tarefas/step.
# Capacidade com Erro (split 6/4) ~= 3.7 tarefas/step.
# ARRIVAL_RATE = 4.0 pune qualquer erro de alocação com crescimento de fila.
ARRIVAL_RATE = 4.0
N_SUBCHANNELS = 10  # Ajustado para permitir a sobrevivência (8 era muito pouco)

# Parâmetros Fixos para Agentes Padrão
FIXED_ALPHA = 0.1
FIXED_GAMMA = 0.9
FIXED_EPSILON = 0.1

# =================================================================================
# 2. SISTEMA FUZZY (MODO "PARANÓICO")
# =================================================================================
def create_fuzzy_system(input_name='td_error'):
    # Variáveis Linguísticas
    queue_usage = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'queue_usage')
    signal_input = ctrl.Antecedent(np.arange(0, 2.1, 0.1), input_name)
    
    alpha = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'alpha')
    temp = ctrl.Consequent(np.arange(0.1, 2.1, 0.1), 'temp')
    panic = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'panic')

    # --- PERTINÊNCIAS ---
    # Gatilho de risco antecipado (0.2 a 0.3)
    queue_usage['safe'] = fuzz.trapmf(queue_usage.universe, [0, 0, 0.2, 0.3])
    queue_usage['risk'] = fuzz.trimf(queue_usage.universe, [0.2, 0.5, 0.8])
    queue_usage['critical'] = fuzz.trapmf(queue_usage.universe, [0.6, 0.8, 1.0, 1.0])

    signal_input['low'] = fuzz.zmf(signal_input.universe, 0.2, 0.6)
    signal_input['high'] = fuzz.smf(signal_input.universe, 0.2, 0.6)

    alpha['low'] = fuzz.trimf(alpha.universe, [0, 0.1, 0.3])
    alpha['high'] = fuzz.trimf(alpha.universe, [0.7, 0.9, 1.0])

    temp['low'] = fuzz.trimf(temp.universe, [0.1, 0.2, 0.4]) 
    temp['high'] = fuzz.trimf(temp.universe, [0.8, 1.2, 1.5]) 

    panic['off'] = fuzz.zmf(panic.universe, 0.3, 0.5)
    panic['on'] = fuzz.smf(panic.universe, 0.3, 0.5)

    # --- REGRAS ---
    rule1 = ctrl.Rule(queue_usage['critical'], (panic['on'], alpha['high'], temp['low']))
    rule2 = ctrl.Rule(queue_usage['risk'], (panic['off'], alpha['high'], temp['low'])) 
    rule3 = ctrl.Rule(signal_input['high'] & queue_usage['safe'], (panic['off'], alpha['high'], temp['high']))
    rule4 = ctrl.Rule(signal_input['low'] & queue_usage['safe'], (panic['off'], alpha['low'], temp['low']))

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(system)

# Instâncias independentes para cada tipo de agente
fuzzy_td_sim = create_fuzzy_system('td_error')
fuzzy_bayesian_sim = create_fuzzy_system('bayesian_surprise')
fuzzy_lev_sim = create_fuzzy_system('levenshtein_dist')

# =================================================================================
# 3. AMBIENTE SIDELINK (AJUSTADO PARA SIMETRIA E CAPACIDADE)
# =================================================================================
class SidelinkEnvironment:
    def __init__(self):
        # VOLTAMOS PARA A SIMETRIA: [2.0, 0.2, 0.2]
        # Se usarmos 0.3, a capacidade desse braço cai abaixo de 4.0 e a fila cresce mesmo com split perfeito.
        self.base_latencies = np.array([2.0, 0.2, 0.2])
        
        # 10 Subcanais = Zona de Oportunidade. 
        # (Split 5/5 -> Load 0.4 -> Latência Baixa -> Capacidade > 4.0)
        self.N_SUBCHANNELS = 10
        self.SENSING_EFFICIENCY = 2.0 

    def drift_step(self, t):
        # Swap de canais a cada 250 steps (Alta Dinamicidade)
        # O Standard IQL vai demorar a perceber e causará aglomeração no canal errado.
        if t > 0 and t % 250 == 0:
            self.base_latencies[1], self.base_latencies[2] = self.base_latencies[2], self.base_latencies[1]
        
        noise = np.random.uniform(-0.0001, 0.0001, size=3)
        self.base_latencies = np.clip(self.base_latencies + noise, 0.1, 3.0)

    def get_latencies(self, actions):
        realized_latencies = np.zeros(N_PLAYERS)
        actions = np.array(actions)
        unique, counts = np.unique(actions, return_counts=True)
        load_map = dict(zip(unique, counts))

        for i in range(N_PLAYERS):
            act = actions[i]
            latency = self.base_latencies[act]
            
            if act != 0: 
                n_users = load_map.get(act, 0)
                # Cálculo de Carga (Baseado em 10 subcanais)
                contention_load = max(0, (n_users - 1)) / self.N_SUBCHANNELS
                
                # Física de Colisão
                p_collision = 1.0 - np.exp(-self.SENSING_EFFICIENCY * contention_load)
                p_collision = min(p_collision, 0.95)
                
                # Latência Efetiva
                congestion_factor = 1.0 / (1.0 - p_collision)
                latency = latency * congestion_factor
                
            realized_latencies[i] = latency
        return realized_latencies

    def calculate_jain_index(self, actions):
        rsus = [a for a in actions if a != 0]
        if not rsus: return 1.0
        count1 = rsus.count(1)
        count2 = rsus.count(2)
        loads = np.array([count1, count2]) + 1e-9 
        sum_x = np.sum(loads)
        sum_x_sq = np.sum(loads**2)
        return (sum_x ** 2) / (2 * sum_x_sq)

# =================================================================================
# 4. AGENTES
# =================================================================================
class Agent:
    def __init__(self): self.current_queue = 0.0
    def get_state(self): return min(int((self.current_queue / MAX_QUEUE) * 10), 10)
    def update_queue_physics(self, processed):
        self.current_queue += ARRIVAL_RATE
        self.current_queue -= processed
        self.current_queue = max(0.0, min(self.current_queue, MAX_QUEUE))
    def select_action(self): raise NotImplementedError
    def update(self, a, r, p): raise NotImplementedError

class RandomAgent(Agent):
    def select_action(self): return random.randint(0, N_ARMS - 1)
    def update(self, a, r, p): self.update_queue_physics(p)

class StandardIQLAgent(Agent):
    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((11, N_ARMS))
    def select_action(self):
        if random.random() < FIXED_EPSILON: return random.randint(0, N_ARMS - 1)
        state = self.get_state()
        q_row = self.q_table[state]
        return np.random.choice(np.flatnonzero(q_row == q_row.max()))
    def update(self, action, reward, processed):
        s = self.get_state()
        self.update_queue_physics(processed)
        next_s = self.get_state()
        target = reward + FIXED_GAMMA * np.max(self.q_table[next_s])
        self.q_table[s, action] += FIXED_ALPHA * (target - self.q_table[s, action])

class SWUCBAgent(Agent):
    def __init__(self, window_size=100):
        super().__init__()
        self.window_size = window_size
        self.history = []
        self.exploration_c = 1.0
    def select_action(self):
        current_window = self.history[-self.window_size:]
        counts = np.zeros(N_ARMS)
        rewards = np.zeros(N_ARMS)
        for a, r in current_window:
            counts[a] += 1
            rewards[a] += r 
        ucb_values = np.zeros(N_ARMS)
        total_counts = len(current_window)
        if total_counts == 0: return random.randint(0, N_ARMS-1)
        for a in range(N_ARMS):
            if counts[a] == 0: ucb_values[a] = 9999
            else:
                avg = rewards[a] / counts[a]
                bonus = self.exploration_c * np.sqrt(np.log(total_counts) / counts[a])
                ucb_values[a] = avg + bonus
        return np.argmax(ucb_values)
    def update(self, action, reward, processed):
        self.update_queue_physics(processed)
        self.history.append((action, reward))
        if len(self.history) > self.window_size * 2: self.history = self.history[-self.window_size:]

class FuzzyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((11, N_ARMS))
        self.last_td_error = 0.0

    def select_action(self):
        queue_pct = self.current_queue / MAX_QUEUE
        error_input = min(abs(self.last_td_error), 2.0)
        
        fuzzy_td_sim.input['queue_usage'] = queue_pct
        fuzzy_td_sim.input['td_error'] = error_input
        try:
            fuzzy_td_sim.compute()
            temp = fuzzy_td_sim.output['temp']
            panic_score = fuzzy_td_sim.output['panic']
        except: 
            temp, panic_score = 0.5, 0.0

        state = self.get_state()
        q_values = self.q_table[state]
        
        # GATILHO DE PÂNICO
        if panic_score > 0.4: 
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))
        
        try:
            exp_q = np.exp((q_values - np.max(q_values)) / temp)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(N_ARMS), p=probs)
        except:
            return np.random.choice(range(N_ARMS))

    def update(self, action, reward, processed):
        s = self.get_state()
        self.update_queue_physics(processed)
        next_s = self.get_state()
        
        target = reward + FIXED_GAMMA * np.max(self.q_table[next_s])
        error = target - self.q_table[s, action]
        self.last_td_error = error
        
        queue_pct = self.current_queue / MAX_QUEUE
        error_input = min(abs(error), 2.0)
        
        fuzzy_td_sim.input['queue_usage'] = queue_pct
        fuzzy_td_sim.input['td_error'] = error_input
        try: 
            fuzzy_td_sim.compute()
            alpha = fuzzy_td_sim.output['alpha']
        except: 
            alpha = 0.1
            
        self.q_table[s, action] += alpha * error

class BayesianFuzzyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((11, N_ARMS))
        # Parâmetros da crença (Prior Beta) para cada braço
        self.beliefs = np.ones((N_ARMS, 2)) # [alpha, beta] iniciados em 1 (Uniforme)
        self.last_surprise = 0.0
        self.latency_threshold = 0.5 # Limiar para definir "sucesso" no Sidelink

    def calculate_kl_beta(self, a1, b1, a2, b2):
        # KL Divergence entre Beta(a1, b1) e Beta(a2, b2)
        # S_R18, S_R97: Medida de "Surpresa" informacional
        term1 = betaln(a2, b2) - betaln(a1, b1)
        term2 = (a1 - a2) * psi(a1)
        term3 = (b1 - b2) * psi(b1)
        term4 = (a2 - a1 + b2 - b1) * psi(a1 + b1)
        return term1 + term2 + term3 + term4

    def select_action(self):
        # Meta-ajuste via Surpresa Bayesiana
        fuzzy_bayesian_sim.input['queue_usage'] = self.current_queue / MAX_QUEUE
        # Escalonamento da surpresa para o universo [0, 2.0] do fuzzy
        fuzzy_bayesian_sim.input['bayesian_surprise'] = min(self.last_surprise * 5.0, 2.0) 
        try:
            fuzzy_bayesian_sim.compute()
            temp = fuzzy_bayesian_sim.output['temp']
            panic_score = fuzzy_bayesian_sim.output['panic']
        except: 
            temp, panic_score = 0.5, 0.0
        
        state = self.get_state()
        q_values = self.q_table[state]

        if panic_score > 0.4: 
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))
        
        try:
            exp_q = np.exp((q_values - np.max(q_values)) / temp)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(N_ARMS), p=probs)
        except:
            return np.random.choice(range(N_ARMS))

    def update(self, action, reward, processed):
        s_old = self.get_state()
        
        # 1. Observação do evento (Binário: Latência baixa ou alta)
        # S_R42: Surpresa baseada no desvio estrutural dos resultados
        success = 1 if abs(reward) < self.latency_threshold else 0
        
        # 2. Calcular Surpresa (KL entre Posterior e Prior) antes de atualizar crença
        a_prior, b_prior = self.beliefs[action]
        a_post, b_post = a_prior + success, b_prior + (1 - success)
        
        self.last_surprise = self.calculate_kl_beta(a_post, b_post, a_prior, b_prior)
        self.beliefs[action] = [a_post, b_post] # Atualiza crença
        
        # 3. Atualização Q-Table com Alpha modulado pela Surpresa
        self.update_queue_physics(processed)
        target = reward + FIXED_GAMMA * np.max(self.q_table[self.get_state()])
        error = target - self.q_table[s_old, action]
        
        fuzzy_bayesian_sim.input['queue_usage'] = self.current_queue / MAX_QUEUE
        fuzzy_bayesian_sim.input['bayesian_surprise'] = min(self.last_surprise * 5.0, 2.0)
        try: 
            fuzzy_bayesian_sim.compute() 
            alpha = fuzzy_bayesian_sim.output['alpha']
        except: 
            alpha = 0.1
        
        self.q_table[s_old, action] += alpha * error

class LevenshteinAgent(Agent):
    def __init__(self, history_length=20):
        super().__init__()
        self.q_table = np.zeros((11, N_ARMS))
        self.history_length = history_length
        self.action_history = []
        self.last_levenshtein_score = 0.0

    def calculate_levenshtein(self, seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x): matrix[x, 0] = x
        for y in range(size_y): matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix[x,y] = matrix[x-1, y-1]
                else:
                    matrix[x,y] = min(
                        matrix[x-1, y] + 1,   # Deletion
                        matrix[x-1, y-1] + 1, # Substitution
                        matrix[x, y-1] + 1    # Insertion
                    )
        # Retorna a distância normalizada pelo tamanho da sequência
        return matrix[size_x-1, size_y-1] / max(len(seq1), 1)

    def select_action(self):
        # 1. Configura a entrada do Fuzzy com a "Surpresa" baseada em Levenshtein
        queue_pct = self.current_queue / MAX_QUEUE
        
        # Levenshtein mede o quanto o comportamento mudou. 
        # Se 0.0 -> Comportamento idêntico (Estável). 
        # Se 1.0 -> Comportamento totalmente diferente (Instável).
        # Multiplicamos por 2.0 para escalar para o universo [0, 2.0] do fuzzy.
        change_intensity = min(self.last_levenshtein_score * 2.0, 2.0)

        fuzzy_lev_sim.input['queue_usage'] = queue_pct
        fuzzy_lev_sim.input['levenshtein_dist'] = change_intensity
        
        try:
            fuzzy_lev_sim.compute()
            temp = fuzzy_lev_sim.output['temp']
            panic_score = fuzzy_lev_sim.output['panic']
        except:
            temp, panic_score = 0.5, 0.0

        state = self.get_state()
        q_values = self.q_table[state]

        if panic_score > 0.4:
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

        try:
            exp_q = np.exp((q_values - np.max(q_values)) / temp)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(N_ARMS), p=probs)
        except:
            return np.random.choice(range(N_ARMS))

    def update(self, action, reward, processed):
        s = self.get_state()
        self.update_queue_physics(processed)
        next_s = self.get_state()
        
        # Atualiza histórico de ações
        self.action_history.append(action)
        if len(self.action_history) > self.history_length:
            # Compara a metade mais recente com a metade mais antiga para detectar mudanças de padrão
            mid = len(self.action_history) // 2
            old_seq = self.action_history[:mid]
            new_seq = self.action_history[mid:]
            self.last_levenshtein_score = self.calculate_levenshtein(old_seq, new_seq)
            self.action_history.pop(0)
        
        # Q-Learning Padrão com Alpha modulado pelo Fuzzy (Levenshtein)
        target = reward + FIXED_GAMMA * np.max(self.q_table[next_s])
        error = target - self.q_table[s, action]
        
        queue_pct = self.current_queue / MAX_QUEUE
        change_intensity = min(self.last_levenshtein_score * 2.0, 2.0)

        fuzzy_lev_sim.input['queue_usage'] = queue_pct
        fuzzy_lev_sim.input['levenshtein_dist'] = change_intensity
        try:
            fuzzy_lev_sim.compute()
            alpha = fuzzy_lev_sim.output['alpha']
        except:
            alpha = 0.1
            
        self.q_table[s, action] += alpha * error

# =================================================================================
# 5. LOOP DE SIMULAÇÃO
# =================================================================================
def run_simulation(agent_type):
    env = SidelinkEnvironment()
    agents = []
    
    for _ in range(N_PLAYERS):
        if agent_type == "Standard": agents.append(StandardIQLAgent())
        elif agent_type == "Fuzzy": agents.append(FuzzyAgent())
        elif agent_type == "Bayesian": agents.append(BayesianFuzzyAgent())
        elif agent_type == "Levenshtein": agents.append(LevenshteinAgent())
        elif agent_type == "Random": agents.append(RandomAgent())
        elif agent_type == "SW-UCB": agents.append(SWUCBAgent())
    
    hq, hl, hf = [], [], []
    
    for t in range(N_STEPS):
        env.drift_step(t)
        actions = [ag.select_action() for ag in agents]
        latencies = env.get_latencies(actions)
        fairness = env.calculate_jain_index(actions)
        
        sq, sl = 0, 0
        for i in range(N_PLAYERS):
            q_pen = 50.0 if agents[i].current_queue > MAX_QUEUE * 0.9 else 0.0
            reward = -(latencies[i] + q_pen)
            
            capacity = 1.0 if actions[i] == 0 else (2.0 / latencies[i])
            
            agents[i].update(actions[i], reward, capacity)
            sq += agents[i].current_queue
            sl += latencies[i]
            
        hq.append(sq/N_PLAYERS)
        hl.append(sl/N_PLAYERS)
        hf.append(fairness)
        
    return hq, hl, hf

# =================================================================================
# 6. EXECUÇÃO
# =================================================================================
# =================================================================================
# 6. EXECUÇÃO
# =================================================================================
algorithms = ["Random", "SW-UCB", "Standard", "Fuzzy", "Bayesian", "Levenshtein"]
N_TRIALS = 33
results = {algo: {'q': [], 'l': [], 'f': []} for algo in algorithms}

print(f"Iniciando Survivor Test (Carga: {ARRIVAL_RATE}, Subcanais: {N_SUBCHANNELS}, Steps: {N_STEPS})...")
print(f"Executando {N_TRIALS} rodadas para cada algoritmo (Alta Confiabilidade Estatística)...")

for algo in algorithms:
    print(f"\nAlgoritmo: {algo}")
    for i in range(N_TRIALS):
        # Seed diferente para cada rodada para garantir independência estatística
        # Mas reprodutível entre algoritmos (embora eles rodem em loops separados aqui)
        np.random.seed(i)
        random.seed(i)
        
        hq, hl, hf = run_simulation(algo)
        results[algo]['q'].append(hq)
        results[algo]['l'].append(hl)
        results[algo]['f'].append(hf)
        print(f".", end="", flush=True)

# --- Estatísticas ---
print("\n\n=== RESULTADOS FINAIS (Média de 33 Rodadas) ===")
stats_data = []
warmup = 100

for algo in algorithms:
    # Converte lista de listas em matriz numpy [N_TRIALS, N_STEPS]
    q_matrix = np.array(results[algo]['q'])
    l_matrix = np.array(results[algo]['l'])
    f_matrix = np.array(results[algo]['f'])
    
    # Médias ao longo do tempo (eixo 1) para cada trial, depois média dos trials
    avg_q = np.mean(q_matrix[:, warmup:])
    max_q = np.mean(np.max(q_matrix[:, warmup:], axis=1)) # Média dos picos de fila
    avg_l = np.mean(l_matrix[:, warmup:])
    avg_f = np.mean(f_matrix[:, warmup:])
    
    stats_data.append({
        "Algorithm": algo,
        "Avg Queue": avg_q,
        "Max Queue (Risk)": max_q,
        "Avg Latency": avg_l,
        "Avg Fairness": avg_f
    })

df_stats = pd.DataFrame(stats_data)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_stats.round(4))

df_stats.to_csv("sidelink_stress_results.csv", index=False)

# --- Gráficos (Com Intervalo de Confiança) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
W = 50 

colors = {'Random': 'gray', 'SW-UCB': 'orange', 'Standard': 'blue', 
          'Fuzzy': 'green', 'Bayesian': 'purple', 'Levenshtein': 'brown'}

for algo in algorithms:
    q_matrix = np.array(results[algo]['q'])
    l_matrix = np.array(results[algo]['l'])
    f_matrix = np.array(results[algo]['f'])
    
    # Função auxiliar para plotar com sombra
    def plot_with_ci(ax, data_matrix, label, color):
        # Suavização
        df = pd.DataFrame(data_matrix.T)
        smooth_df = df.rolling(W).mean()
        
        mean_series = smooth_df.mean(axis=1)
        std_series = smooth_df.std(axis=1)
        
        # Intervalo de Confiança 95% (1.96 * erro padrão)
        ci = 1.96 * std_series / np.sqrt(N_TRIALS)
        
        x = range(len(mean_series))
        ax.plot(x, mean_series, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean_series - ci, mean_series + ci, color=color, alpha=0.2)
        
    plot_with_ci(axes[0], q_matrix, algo, colors.get(algo, 'black'))
    plot_with_ci(axes[1], l_matrix, algo, colors.get(algo, 'black'))
    plot_with_ci(axes[2], f_matrix, algo, colors.get(algo, 'black'))

axes[0].set_title("Fila Média (IC 95%)")
axes[0].set_ylabel("Pacotes na Fila")
axes[0].set_xlabel("Steps")
axes[0].axhline(y=MAX_QUEUE, color='r', ls='--', label="Capacidade Máx")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Latência Efetiva (IC 95%)")
axes[1].set_ylabel("Segundos (Normalizado)")
axes[1].set_xlabel("Steps")
axes[1].grid(True, alpha=0.3)

axes[2].set_title("Índice de Jain (IC 95%)")
axes[2].set_ylabel("Índice (0-1)")
axes[2].set_xlabel("Steps")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fuzzy_vs_standard_stress_test.png", dpi=300)
print("Plot saved to 'fuzzy_vs_standard_stress_test.png'")