
# =================================================================================
# AGENTE BASEADO EM DISTÂNCIA DE LEVENSHTEIN (EDIT OPERATION MEASURES)
# =================================================================================
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
