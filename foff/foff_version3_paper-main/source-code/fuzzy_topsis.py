import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List
from enum import Enum, auto

class Criteria(Enum):
    BENEFIT = auto()
    COST = auto()

def fuzzify(value, fuzzy_sets_tuple):
    """Fuzzifica um valor de acordo com os conjuntos fuzzy fornecidos."""
    max_membership = 0
    ling_var, lv_dict = fuzzy_sets_tuple
    first_value = list(lv_dict.keys())[0]
    linguistic_term = first_value
    a, b, c = lv_dict[linguistic_term]
    triangular_fuzzy_number = np.array([a, b, c])
    for term_name in ling_var.terms:
        term = ling_var[term_name]
        membership = fuzz.interp_membership(ling_var.universe, term.mf, value)
        
        if membership > max_membership:
            max_membership = membership
            linguistic_term = term_name
            a, b, c = lv_dict[term_name]
            triangular_fuzzy_number = np.array([a, b, c])
    return linguistic_term, triangular_fuzzy_number

def normalize_by_cost_vector(D):
    """Normaliza a matriz D usando normalização vetorial para critérios de custo."""
    D_normalized = np.zeros_like(D, dtype=float)
    for k in range(D.shape[1]):  # Itera pelos critérios
        sum_squares = np.sum(D[:, k, :]**2)
        sqrt_sum_squares = np.sqrt(sum_squares)
        if sqrt_sum_squares == 0:  # Evita divisão por zero
            D_normalized[:, k, :] = 0
        else:
            D_normalized[:, k, :] = D[:, k, :] / sqrt_sum_squares
    return D_normalized

# def normalize_by_cost(D):
#     """Normaliza a matriz D de acordo com os princípios do TOPSIS para números fuzzy
#     triangulares usando o método comum (inverso) para critérios de custo.
#     """
#     D_normalized = np.zeros_like(D, dtype=float)
#     for k in range(D.shape[1]):  # Itera pelos critérios (time, energy)
#         a_min = np.inf  # Inicializa a_min com infinito positivo
#         for i in range(D.shape[0]):
#             a_min = min(a_min, D[i, k, 0])  # Encontra o menor valor 'a' para o critério atual

#         for i in range(D.shape[0]):
#             a, b, c = D[i, k, 0], D[i, k, 1], D[i, k, 2]
#             # Evita divisão por zero se a_min for zero
#             if a_min == 0:
#                 D_normalized[i, k, 0] = 0.0
#                 D_normalized[i, k, 1] = 0.0
#                 D_normalized[i, k, 2] = 0.0
#             else:
#                 D_normalized[i, k, 0] = a_min / c
#                 D_normalized[i, k, 1] = a_min / b
#                 D_normalized[i, k, 2] = a_min / a

#     return D_normalized

# def normalize_by_cost(D):
#     """
#     Normaliza a matriz D para critérios de custo com estabilidade numérica
#     """
#     D_normalized = np.zeros_like(D, dtype=float)
#     for k in range(D.shape[1]):
#         a_min = np.min(D[:, k, 0])
#         c_max = np.max(D[:, k, 2])
        
#         if c_max == 0 or a_min == 0:
#             D_normalized[:, k, :] = 0.0
#         else:
#             for i in range(D.shape[0]):
#                 a, b, c = D[i, k, :]
#                 D_normalized[i, k, 0] = a_min / c_max
#                 D_normalized[i, k, 1] = a_min / b
#                 D_normalized[i, k, 2] = a_min / a
#     return D_normalized

def normalize_by_cost(D):
    """Normaliza a matriz D para critérios de custo com estabilidade numérica usando np.vectorize"""
    a_min = np.min(D[:, :, 0], axis=0)
    c_max = np.max(D[:, :, 2], axis=0)
    
    # Prevenir divisão por zero
    valid_mask = (c_max != 0) & (a_min != 0)
    
    def normalize_element(a, b, c, a_min, c_max, valid):
        if not valid:
            return [0.0, 0.0, 0.0]
        return [a_min / c_max, a_min / b, a_min / a]

    vectorized_function = np.vectorize(normalize_element, otypes=[np.ndarray])
    D_normalized = vectorized_function(D[:, :, 0], D[:, :, 1], D[:, :, 2], a_min, c_max, valid_mask)

    return np.array(D_normalized.tolist())

# def normalize_by_benefit_vector(D):
#     """Normaliza a matriz D usando normalização vetorial para critérios de benefício."""
#     D_normalized = np.zeros_like(D, dtype=float)
#     for k in range(D.shape[1]):  # Itera pelos critérios
#         sum_squares = np.sum(D[:, k, :]**2)
#         sqrt_sum_squares = np.sqrt(sum_squares)
#         if sqrt_sum_squares == 0:  # Evita divisão por zero
#             D_normalized[:, k, :] = 0
#         else:
#             D_normalized[:, k, :] = D[:, k, :] / sqrt_sum_squares
#     return D_normalized

def normalize_by_benefit(D):
    """
    Normaliza a matriz D de acordo com os princípios do TOPSIS para números fuzzy
    triangulares usando o método comum para critérios de benefício.
    """
    D_normalized = np.zeros_like(D, dtype=float)
    for k in range(D.shape[1]):  # Itera pelos critérios
        c_max = 0
        for i in range(D.shape[0]):
            c_max = max(c_max, D[i, k, 2])  # Encontra o maior valor 'c' para o critério atual

        for i in range(D.shape[0]):
            a, b, c = D[i, k, 0], D[i, k, 1], D[i, k, 2]
            # Evita divisão por zero se c_max for zero
            if c_max == 0:
                D_normalized[i, k, 0] = 0.0
                D_normalized[i, k, 1] = 0.0
                D_normalized[i, k, 2] = 0.0
            else:
                D_normalized[i, k, 0] = a / c_max
                D_normalized[i, k, 1] = b / c_max
                D_normalized[i, k, 2] = c / c_max

    return D_normalized

def convert_dm_to_tfn(D_tilde, fuzzy_sets_tuple):
    result = np.zeros((D_tilde.shape[0], D_tilde.shape[1], 3), dtype=float)
    fuzzy_dict = fuzzy_sets_tuple[1]
    for i in range(D_tilde.shape[0]):
        for j in range(D_tilde.shape[1]):
            term_name = D_tilde[i, j]
            if term_name in fuzzy_dict:
                result[i, j, :] = fuzzy_dict[term_name]  # Atribui o TFN
            else:
                print(f"Termo '{term_name}' não encontrado no dicionário fuzzy.")
                result[i, j, :] = [0, 0, 0]  # TFN padrão ou outro tratamento de erro

    return result


def convert_to_tfn(list_of_terms: List[str], fuzzy_sets_tuple):
    result = []
    fuzzy_dict = fuzzy_sets_tuple[1]
    for term_name in list_of_terms:
        result.append(fuzzy_dict[term_name])
    return np.array(result)

def euclidean_distance(a, b):
 return np.linalg.norm(np.array(a) - np.array(b))

def sum_squared_difference_tfn(tfn1, tfn2):
    return (tfn1[0] - tfn2[0])**2 + (tfn1[1] - tfn2[1])**2 + (tfn1[2] - tfn2[2])**2

# def positive_ideal_solution(weighted_normalized_decision_matrix, criteria_types: List[Criteria]):
#     V_plus = None
#     for j in range(weighted_normalized_decision_matrix.shape[1]):  # Para cada critério
#         if criteria_types[j] == Criteria.BENEFIT:
#             V_plus = np.ones((weighted_normalized_decision_matrix.shape[1], weighted_normalized_decision_matrix.shape[2]))
#         elif criteria_types[j] == Criteria.COST:
#             V_plus = np.zeros((weighted_normalized_decision_matrix.shape[1], weighted_normalized_decision_matrix.shape[2]))
#     return V_plus

# def negative_ideal_solution(weighted_normalized_decision_matrix, criteria_types: List[Criteria]):
#     V_minus = None
#     for j in range(weighted_normalized_decision_matrix.shape[1]):  # Para cada critério
#         if criteria_types[j] == Criteria.BENEFIT:
#             V_minus = np.zeros((weighted_normalized_decision_matrix.shape[1], weighted_normalized_decision_matrix.shape[2]))
#         elif criteria_types[j] == Criteria.COST:
#             V_minus = np.ones((weighted_normalized_decision_matrix.shape[1], weighted_normalized_decision_matrix.shape[2]))
#     return V_minus

def positive_ideal_solution(V_tilde, criteria_types):
    num_criteria = V_tilde.shape[1]
    A_plus = np.zeros((num_criteria, 3))
    for j in range(num_criteria):
        if criteria_types[j] == Criteria.BENEFIT:
            A_plus[j, 0] = np.max(V_tilde[:, j, 2])  # max a
            A_plus[j, 1] = np.max(V_tilde[:, j, 2])  # max b
            A_plus[j, 2] = np.max(V_tilde[:, j, 2])  # max c
        else:
            A_plus[j, 0] = np.min(V_tilde[:, j, 0])  # min a
            A_plus[j, 1] = np.min(V_tilde[:, j, 0])  # min b
            A_plus[j, 2] = np.min(V_tilde[:, j, 0])  # min c
    return A_plus

def negative_ideal_solution(V_tilde, criteria_types):
    num_criteria = V_tilde.shape[1]
    A_minus = np.zeros((num_criteria, 3))
    for j in range(num_criteria):
        if criteria_types[j] == Criteria.BENEFIT:
            A_minus[j, 0] = np.min(V_tilde[:, j, 0])  # min a
            A_minus[j, 1] = np.min(V_tilde[:, j, 0])  # min b
            A_minus[j, 2] = np.min(V_tilde[:, j, 0])  # min c
        else:
            A_minus[j, 0] = np.max(V_tilde[:, j, 2])  # max a
            A_minus[j, 1] = np.max(V_tilde[:, j, 2])  # max b
            A_minus[j, 2] = np.max(V_tilde[:, j, 2])  # max c
    return A_minus

def calculate_distances(V_tilde, A_plus, A_minus):
    """
    Calcula as distâncias D+ e D- para cada alternativa usando a distância euclidiana para TFNs.
    """
    num_alternatives = V_tilde.shape[0]
    num_criteria = V_tilde.shape[1]
    D_plus = np.zeros(num_alternatives)
    D_minus = np.zeros(num_alternatives)

    for i in range(num_alternatives):
        for j in range(num_criteria):
            D_plus[i]  += np.sqrt(sum_squared_difference_tfn(V_tilde[i, j, :], A_plus[j, :])/3.0)
            D_minus[i] += np.sqrt(sum_squared_difference_tfn(V_tilde[i, j, :], A_minus[j, :])/3.0)

    return D_plus, D_minus

def calculate_closeness_coefficient(D_plus, D_minus):
  """Calcula o coeficiente de proximidade (CCi) para cada alternativa."""
  return D_minus / (D_plus + D_minus)

def fuzzy_topsis(D_tilde, W_tilde, criteria_types: List[Criteria]):
    R_tilde = np.zeros_like(D_tilde, dtype=float)
    
    for j in range(D_tilde.shape[1]):
        if criteria_types[j] == Criteria.BENEFIT:
            R_tilde[:, j, :] = normalize_by_benefit(D_tilde[:, [j], :])[:, 0, :]
        elif criteria_types[j] == Criteria.COST:
            R_tilde[:, j, :] = normalize_by_cost(D_tilde[:, [j], :])[:, 0, :]
        else:
            raise ValueError(f"Invalid criteria type for criterion {j}: {criteria_types[j]}")
    V_tilde = R_tilde * W_tilde
    # Soluções ideais
    A_plus = positive_ideal_solution(V_tilde, criteria_types)
    A_minus = negative_ideal_solution(V_tilde, criteria_types)
    # Calcular distâncias D+ e D-
    D_plus, D_minus = calculate_distances(V_tilde, A_plus, A_minus)
    # Calcular coeficiente de proximidade (CCi)
    CCi = calculate_closeness_coefficient(D_plus, D_minus)
    return CCi
