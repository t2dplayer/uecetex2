import numpy as np
import skfuzzy as fuzz
from typing import List
from enum import Enum, auto

class Criteria(Enum):
    BENEFIT = auto()
    COST = auto()

def sum_squared_difference_tfn(tfn1, tfn2):
    return (tfn1[0] - tfn2[0])**2 + (tfn1[1] - tfn2[1])**2 + (tfn1[2] - tfn2[2])**2

def normalize_by_cost(D):
    a_min = np.min(D[:, :, 0], axis=0)
    c_max = np.max(D[:, :, 2], axis=0)
    
    valid_mask = (c_max != 0) & (a_min != 0)
    
    def normalize_element(a, b, c, a_min, c_max, valid):
        if not valid:
            return [0.0, 0.0, 0.0]
        # Avoid division by zero if a, b, or c are 0 (though less likely in valid mask context if designed right)
        # Standard Fuzzy TOPSIS cost normalization: (amin/c, amin/b, amin/a)
        # We need to be careful about 0.
        if c == 0 or b == 0 or a == 0:
             return [0.0, 0.0, 0.0]
        return [a_min / c, a_min / b, a_min / a]

    # Iterate manually to be safe or vectorize carefully. Manual for clarity/safety here.
    D_norm = np.zeros_like(D, dtype=float)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if valid_mask[j]:
                a, b, c = D[i, j]
                if a > 0 and b > 0 and c > 0:
                     D_norm[i, j] = [a_min[j]/c, a_min[j]/b, a_min[j]/a]
    return D_norm

def normalize_by_benefit(D):
    c_max = np.max(D[:, :, 2], axis=0)
    valid_mask = (c_max != 0)
    
    D_norm = np.zeros_like(D, dtype=float)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if valid_mask[j]:
                a, b, c = D[i, j]
                D_norm[i, j] = [a/c_max[j], b/c_max[j], c/c_max[j]]
    return D_norm

def positive_ideal_solution(V_tilde, criteria_types):
    num_criteria = V_tilde.shape[1]
    A_plus = np.zeros((num_criteria, 3))
    for j in range(num_criteria):
        # FPIS: Maximize Benefit (max c, max c, max c), Minimize Cost (min a, min a, min a) ??
        # After normalization, everything is 'benefit-like' usually in vector normalization,
        # but in Fuzzy Cost Norm (amin/c, ...), the values are effectively benefit-like (higher is closer to ideal).
        # Let's assume standard Chen's method:
        # After normalization, v_tilde is compatible.
        # FPIS: v* = (1,1,1) if normalized to [0,1].
        # OR max(v_tilde)
        col = V_tilde[:, j]
        max_c = np.max(col[:, 2])
        A_plus[j] = [max_c, max_c, max_c]
    return A_plus

def negative_ideal_solution(V_tilde, criteria_types):
    num_criteria = V_tilde.shape[1]
    A_minus = np.zeros((num_criteria, 3))
    for j in range(num_criteria):
        # FNIS: min a
        col = V_tilde[:, j]
        min_a = np.min(col[:, 0])
        A_minus[j] = [min_a, min_a, min_a]
    return A_minus

def calculate_distances(V_tilde, A_plus, A_minus):
    num_alternatives = V_tilde.shape[0]
    num_criteria = V_tilde.shape[1]
    D_plus = np.zeros(num_alternatives)
    D_minus = np.zeros(num_alternatives)

    for i in range(num_alternatives):
        for j in range(num_criteria):
            D_plus[i]  += np.sqrt(sum_squared_difference_tfn(V_tilde[i, j], A_plus[j])/3.0)
            D_minus[i] += np.sqrt(sum_squared_difference_tfn(V_tilde[i, j], A_minus[j])/3.0)
    return D_plus, D_minus

def fuzzy_topsis(D_tilde, W_tilde, criteria_types: List[Criteria]):
    """
    D_tilde: (Alternatives, Criteria, 3)
    W_tilde: (Criteria, 3)
    """
    # 1. Normalize
    R_tilde = np.zeros_like(D_tilde, dtype=float)
    # Check criteria types
    # Simplified: assume input D_tilde is mixed cost/benefit
    
    # We need to split logic
    for j in range(D_tilde.shape[1]):
        if criteria_types[j] == Criteria.BENEFIT:
             # Extract column j across all alternatives
             col = D_tilde[:, [j], :] # (N, 1, 3)
             norm = normalize_by_benefit(col)
             R_tilde[:, j, :] = norm[:, 0, :]
        else:
             col = D_tilde[:, [j], :] # (N, 1, 3)
             norm = normalize_by_cost(col)
             R_tilde[:, j, :] = norm[:, 0, :]
             
    # 2. Weighted Normalized
    V_tilde = np.zeros_like(R_tilde)
    for i in range(R_tilde.shape[0]):
        for j in range(R_tilde.shape[1]):
            # Fuzzy multiplication: (a,b,c) * (wa, wb, wc) approx (a*wa, ...)
            r = R_tilde[i, j]
            w = W_tilde[j]
            V_tilde[i, j] = [r[0]*w[0], r[1]*w[1], r[2]*w[2]]
            
    # 3. Ideals
    A_plus = positive_ideal_solution(V_tilde, criteria_types)
    A_minus = negative_ideal_solution(V_tilde, criteria_types)
    
    # 4. Distances
    D_plus, D_minus = calculate_distances(V_tilde, A_plus, A_minus)
    
    # 5. Closeness
    with np.errstate(divide='ignore', invalid='ignore'):
        CCi = D_minus / (D_plus + D_minus)
        CCi = np.nan_to_num(CCi) # Handle 0/0
        
    return CCi