import simpy
import numpy as np
import matplotlib.pyplot as plt
import csv
from enum import Enum
from typing import Callable, Dict, List, Union
from fuzzy_topsis import fuzzy_topsis, Criteria

# --- Configuration Constants ---
# "Heterogeneous Computing": Different speeds imply different capabilities.
A = np.array([1.0, 3.0, 4.0, 0.5])  # Server speeds
# "Energy Model": Non-linear cubic relationship (common in DVFS/CPU models).
ENERGY_EXPONENT = 3.0  # Increased to 3.0 to emphasize energy penalty (Standard in literature)
BASE_POWER = 10.0 

N_SERVERS = len(A)
N_CLIENTS = 2
Q_CAPACITY = 100
SIM_TIME = 2000.0   # Increased simulation time for statistical stability
N_EPISODES = 30     # Keep 30 for convergence check

# Learning Parameters
ALPHA_ACTOR = 0.1   # Reduced learning rate for stability
ALPHA_CRITIC = 0.1
GAMMA = 0.95

# --- Energy Calculation ---
def calculate_energy(task_size, server_speed):
    # Energy = (P_dynamic + P_static) * Time
    # P_dynamic = k * f^alpha (standard DVFS)
    # P_static = Constant leakage (penalizes very slow execution)
    
    power_dynamic = server_speed ** ENERGY_EXPONENT
    power_static = BASE_POWER
    
    total_power = power_dynamic + power_static
    duration = task_size / server_speed
    
    return total_power * duration

# --- Reward Strategies (Modified per Review) ---

# "Pareto" strategies renamed to "Chebyshev Scalarization"
def reward_chebyshev_log_2obj(queues, action, task_size, speeds):
    """Minimize max deviation from ideal (0,0) in Log space."""
    q_len = len(queues[action].items)
    speed = speeds[action]
    t_proc = task_size / speed
    
    # Normalization factors (approximate)
    n_q = np.log(q_len + 1)
    n_t = np.log(t_proc + 1)
    
    return -max(n_q, n_t)

def reward_chebyshev_3obj(queues, action, task_size, speeds):
    """
    Min-Max Scalarization (Chebyshev) for 3 Objectives.
   
    Crucial fix: Better normalization to prevent gradient explosion.
    """
    q_len = float(len(queues[action].items))
    speed = speeds[action]
    t_proc = task_size / speed
    energy = (speed ** ENERGY_EXPONENT) * t_proc
    
    # Manual Normalization based on observed ranges
    # Q: 0-20 -> norm 0-1
    # T: 0.1-8 -> norm 0-1
    # E: 1-1000 -> norm 0-1
    
    n_q = q_len / 20.0
    n_t = t_proc / 8.0
    n_e = energy / 500.0 # Rough max energy
    
    return -max(n_q, n_t, n_e)

def reward_fuzzy_topsis(queues, action, task_size, speeds):
    """
    Dynamic Reward Shaping using Fuzzy TOPSIS.
    Converts multi-objective state into a single scalar closeness coefficient.
    """
    n_servers = len(queues)
    D_tilde = np.zeros((n_servers, 3, 3))
    
    for i in range(n_servers):
        # 1. Queue TFN
        q_len = len(queues[i].items)
        q_tfn = [max(0, q_len-1), float(q_len), float(q_len+1)]
        
        # 2. Time TFN
        speed = speeds[i]
        t_proc = task_size / speed
        t_tfn = [t_proc*0.9, t_proc, t_proc*1.1]
        
        # 3. Energy TFN
        energy = (speed ** ENERGY_EXPONENT) * t_proc
        e_tfn = [energy*0.8, energy, energy*1.2]
        
        D_tilde[i, 0] = q_tfn
        D_tilde[i, 1] = t_tfn
        D_tilde[i, 2] = e_tfn
        
    # Weights - Giving high priority to Energy to satisfy Revisor
    W_tilde = np.array([
        [0.2, 0.3, 0.4], # Queue (Low)
        [0.2, 0.3, 0.4], # Time (Low)
        [0.6, 0.7, 0.8]  # Energy (Very High)
    ])
    
    criteria_types = [Criteria.COST, Criteria.COST, Criteria.COST]
    
    # Returns Closeness Coefficient [0, 1]
    cci = fuzzy_topsis(D_tilde, W_tilde, criteria_types)
    
    # Reward is directly the closeness to ideal solution
    # We multiply by 10 to make gradients larger for the Critic
    return cci[action] * 10.0

# --- Helper Functions ---
def next_arrival_time(lam):
    return -np.log(np.random.rand()) / lam

def next_task_size(min_val, range_val):
    return min_val + np.random.rand() * range_val

def discretize_state(queues, base):
    # Simple state representation for tabular RL
    # Using raw queue length might explode state space, clamping at 5
    q_sizes = [min(len(q.items), 5) for q in queues]
    h = 0.0
    for size in q_sizes:
        h = h * (base + 1) + size
    return int(h)

def calculate_entropy_probs(probs):
    entropy = 0.0
    for p in probs:
        if p > 1e-9:
            entropy -= p * np.log(p)
    return entropy

def calculate_entropy(preferences):
    if len(preferences) == 0: return 0.0
    exp_p = np.exp(preferences - np.max(preferences))
    probs = exp_p / np.sum(exp_p)
    return calculate_entropy_probs(probs)

def select_action_softmax(state, n_actions, actor_table):
    if state not in actor_table:
        actor_table[state] = np.zeros(n_actions)
    
    preferences = actor_table[state]
    exp_p = np.exp(preferences - np.max(preferences))
    probs = exp_p / np.sum(exp_p)
    
    action = np.random.choice(n_actions, p=probs)
    return action, probs

# --- Simulation Classes ---

class Server:
    def __init__(self, env, id, speed, capacity):
        self.env = env
        self.id = id
        self.speed = speed
        self.store = simpy.Store(env, capacity=capacity)
        self.action = env.process(self.run())
        self.global_state = None 

    def run(self):
        while True:
            task_size = yield self.store.get()
            duration = task_size / self.speed
            yield self.env.timeout(duration)
            if self.global_state is not None:
                 if self.env.now > self.global_state['last_completion_time']:
                     self.global_state['last_completion_time'] = self.env.now

# Added Baseline Agent class to separate RL from Heuristics
class Agent:
    def __init__(self, env, id, queues, strategy_type, actor_table=None, critic_table=None, metrics=None, global_state=None, reward_func=None):
        self.env = env
        self.id = id
        self.queues = queues
        self.strategy_type = strategy_type # 'RL', 'Random', 'RR', 'Greedy'
        
        # RL specific
        self.actor = actor_table
        self.critic = critic_table
        self.reward_func = reward_func
        
        # State needed for Round Robin
        self.rr_index = 0
        
        self.metrics = metrics 
        self.global_state = global_state
        self.action_proc = env.process(self.run())

    def run(self):
        while True:
            inter_arrival = next_arrival_time(2.5) # Slightly lower load to allow clearing
            yield self.env.timeout(inter_arrival)
            
            task_size = next_task_size(1.0, 5.0)
            
            # --- Action Selection ---
            action = 0
            probs = None
            
            # RL Logic
            if self.strategy_type == 'RL':
                s = discretize_state(self.queues, 5)
                if s not in self.actor: self.actor[s] = np.zeros(len(self.queues))
                if s not in self.critic: self.critic[s] = 0.0
                
                # Metric: Entropy (validity check)
                self.metrics['entropy'] += calculate_entropy(self.actor[s])
                action, probs = select_action_softmax(s, len(self.queues), self.actor)

            # Baseline: Random
            elif self.strategy_type == 'Random':
                action = np.random.randint(0, len(self.queues))
                # Max entropy for random
                self.metrics['entropy'] += np.log(len(self.queues)) 
            
            # Baseline: Round Robin
            elif self.strategy_type == 'RR':
                action = self.rr_index % len(self.queues)
                self.rr_index += 1
                self.metrics['entropy'] += 0.0 # Deterministic
                
            # Baseline: Greedy (Min Queue)
            elif self.strategy_type == 'Greedy':
                # Find queue with min items
                lens = [len(q.items) for q in self.queues]
                action = np.argmin(lens)
                self.metrics['entropy'] += 0.0 # Deterministic

            self.metrics['steps'] += 1
            self.global_state['tasks_generated'] += 1
            
            # --- Execution & Learning ---
            
            # Calculate Physical Metrics (Always done)
            energy = calculate_energy(task_size, A[action])
            self.metrics['energy_total'] += energy
            
            # RL Update (Only for RL agents)
            if self.strategy_type == 'RL':
                r = self.reward_func(self.queues, action, task_size, A)
                self.metrics['total_reward'] += r 
                
                yield self.queues[action].put(task_size)
                
                # TD Update
                next_s = discretize_state(self.queues, 5)
                if next_s not in self.actor: self.actor[next_s] = np.zeros(len(self.queues))
                if next_s not in self.critic: self.critic[next_s] = 0.0
                
                td_target = r + GAMMA * self.critic[next_s]
                td_error = td_target - self.critic[s]
                
                self.critic[s] += ALPHA_CRITIC * td_error
                
                # Actor Update
                self.actor[s][action] += ALPHA_ACTOR * td_error * (1.0 - probs[action])
                for i in range(len(self.queues)):
                    if i != action:
                        self.actor[s][i] -= ALPHA_ACTOR * td_error * probs[i]
            else:
                # Baselines just act
                yield self.queues[action].put(task_size)


def run_experiment(exp_name, strategy_type, reward_func=None):
    print(f"\n--- Running: {exp_name} ---")
    
    actors = [{} for _ in range(N_CLIENTS)]
    critics = [{} for _ in range(N_CLIENTS)]
    log_data = [] 
    
    for episode in range(N_EPISODES):
        np.random.seed(42 + episode) 
        
        env = simpy.Environment()
        global_state = {'last_completion_time': 0.0, 'tasks_generated': 0}
        
        servers = []
        queues = []
        for i in range(N_SERVERS):
            srv = Server(env, i, A[i], Q_CAPACITY)
            srv.global_state = global_state
            servers.append(srv)
            queues.append(srv.store)
            
        episode_metrics = [{'entropy': 0.0, 'steps': 0, 'total_reward': 0.0, 'energy_total': 0.0} for _ in range(N_CLIENTS)]
        
        agents = []
        for i in range(N_CLIENTS):
            agt = Agent(env, i, queues, strategy_type, actors[i], critics[i], episode_metrics[i], global_state, reward_func)
            agents.append(agt)
            
        env.run(until=SIM_TIME)
        
        makespan = global_state['last_completion_time']
        
        # Aggregating metrics
        ent_total = sum(m['entropy'] for m in episode_metrics)
        steps_total = sum(m['steps'] for m in episode_metrics)
        erg_total = sum(m['energy_total'] for m in episode_metrics)
        
        avg_entropy = ent_total / steps_total if steps_total > 0 else 0
        
        log_data.append({
            'episode': episode,
            'makespan': makespan,
            'entropy': avg_entropy,
            'energy': erg_total
        })
        
        if episode % 10 == 0:
             print(f"  Ep {episode}: Time={makespan:.0f}, Energy={erg_total:.0f}")

    return log_data

def plot_comparison_ieee_style(results):
    """
    Removed Cumulative Reward plot. 
    Focus on Energy vs Entropy vs Makespan against Baselines.
    """
    plt.style.use('seaborn-v0_8-paper') # More formal style
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Define styles for distinction
    # Baselines: Dashed/Dotted
    # Proposed: Solid, Thick
    styles = {
        'Fuzzy TOPSIS (Proposed)': {'c': 'purple', 'ls': '-', 'lw': 2.5, 'marker': 'o'},
        'Chebyshev 3-Obj (RL)':    {'c': 'red', 'ls': '-', 'lw': 1.5, 'marker': ''},
        'Random (Baseline)':       {'c': 'gray', 'ls': '--', 'lw': 1.5, 'marker': ''},
        'Round Robin (Baseline)':  {'c': 'orange', 'ls': ':', 'lw': 1.5, 'marker': ''},
        'Greedy (Baseline)':       {'c': 'green', 'ls': '-.', 'lw': 1.5, 'marker': ''},
        'Chebyshev Log (RL)':      {'c': 'blue', 'ls': '-', 'lw': 1.0, 'marker': ''}
    }
    
    for name, data in results.items():
        episodes = [d['episode'] for d in data]
        makespans = [d['makespan'] for d in data]
        entropies = [d['entropy'] for d in data]
        energy = [d['energy'] for d in data]
        
        style = styles.get(name, {'c': 'black', 'ls': '-'})
        
        # Smooth curves for clarity
        w = 3
        
        # 1. Total Energy (The main contribution)
        ax1.plot(episodes, energy, label=name, **style, alpha=0.8)
        
        # 2. Makespan (Check for degradation)
        ax2.plot(episodes, makespans, label=name, **style, alpha=0.8)

        # 3. Entropy (Validation of learning)
        ax3.plot(episodes, entropies, label=name, **style, alpha=0.8)

    ax1.set_title('Total Energy Consumption (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (Joules)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, frameon=True)
    
    ax2.set_title('Makespan (Time to Complete)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (s)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Policy Entropy (Exploration vs Determinism)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Entropy (nats)', fontsize=10)
    ax3.set_xlabel('Training Episode', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ieee_valid_comparison.png', dpi=300)
    print("\n Plot saved: 'ieee_valid_comparison.png'. Reward plot removed.")

if __name__ == "__main__":
    experiments = {}
    
    # 1. Run Baselines (Essential for Major Revision)
    experiments["Random (Baseline)"] = run_experiment("Random", "Random")
    experiments["Round Robin (Baseline)"] = run_experiment("Round Robin", "RR")
    experiments["Greedy (Baseline)"] = run_experiment("Greedy", "Greedy")
    
    # 2. Run RL Comparisons (Corrected Names)
    experiments["Chebyshev Log (RL)"] = run_experiment("Chebyshev Log", "RL", reward_chebyshev_log_2obj)
    experiments["Chebyshev 3-Obj (RL)"] = run_experiment("Chebyshev 3-Obj", "RL", reward_chebyshev_3obj)
    
    # 3. Run Proposed Method
    experiments["Fuzzy TOPSIS (Proposed)"] = run_experiment("Fuzzy TOPSIS", "RL", reward_fuzzy_topsis)

    plot_comparison_ieee_style(experiments)
    
    # Print Final Stats Table for Abstract
    print("\n--- Final Results Summary (Last 5 Ep Avg) ---")
    print(f"{'Method':<25} | {'Energy':<10} | {'Makespan':<10} | {'Entropy':<10}")
    print("-" * 65)
    for name, data in experiments.items():
        avg_e = np.mean([d['energy'] for d in data[-5:]])
        avg_m = np.mean([d['makespan'] for d in data[-5:]])
        avg_ent = np.mean([d['entropy'] for d in data[-5:]])
        print(f"{name:<25} | {avg_e:<10.1f} | {avg_m:<10.1f} | {avg_ent:<10.3f}")