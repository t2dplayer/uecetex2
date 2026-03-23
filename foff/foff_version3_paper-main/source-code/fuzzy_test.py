from fuzzy_topsis import *
import matplotlib.pyplot as plt
import re

def format_filename(text):
    # Remove characters that are not alphanumeric, spaces, or hyphens
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # Replace spaces and hyphens with underscores
    formatted_text = re.sub(r'[\s\-]+', '_', cleaned_text)
    # Convert to lowercase for consistency
    formatted_text = formatted_text.lower()
    return formatted_text

def plot_fuzzy_set(lt_, terms, title, x_label, y_label, xlim_l, xlim_u, ylim_l, ylim_u, loc):
    axs = None
    # fig, axs = FuzzyVariableVisualizer(lt_).view(ax=axs)
    fig, axs = plt.subplots(figsize=(8, 6))

    # Cores para os conjuntos fuzzy
    colors = ['#1f77b4', '#d6604d', '#00a693', '#db94b9', '#f0e442', '#7570b3', '#e7298a']
    legend_handles = []
    legend_labels = []
    # Modificar o estilo das linhas e preenchimento
    for i, term in enumerate(lt_.terms.values()):
        line, = axs.plot(lt_.universe, term.mf, color=colors[i], linestyle='--', linewidth=2, label=str(term))  # Add label here
        axs.fill_between(lt_.universe, term.mf, color=colors[i], alpha=0.2)

        legend_handles.append(line) # Store line object for legend
        legend_labels.append(f'Fuzzy Set ({terms[i]})') # Store corresponding label

    # Personalizar o título e os eixos
    # axs.set_title(title, fontsize=16, fontweight='bold')
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)

    # Personalizar fundo e grid
    axs.set_facecolor('white')  # Fundo branco
    axs.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

    # Ajustar os limites dos eixos
    axs.set_xlim(xlim_l, xlim_u)  # Limites do eixo X
    axs.set_ylim(ylim_l, ylim_u)  # Limites do eixo Y

    # Personalizar a legenda
    axs.legend(
        handles=legend_handles,  # Use handles for accurate colors
        labels=legend_labels,  # Use labels based on input terms
        loc=loc
    )
    plt.tight_layout()
    plt.savefig(f"{format_filename(title)}.pdf", format="pdf")
    plt.show()

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

def make_weights_matrix(W, weights, function):
    W_tilde = np.zeros(shape=(W.shape[0], 3))
    for i in range(W.shape[0]):
        memberships = np.array(
            [
                fuzz.interp_membership(
                    weights.universe, weights[term].mf, 
                    W[i]
                ) for term in weights
            ]
        )
        a, b, c = function(memberships, weights)
        W_tilde[i] = np.array([a, b, c])
    return W_tilde

def make_decision_matrix(D, ratings, function):
    D_tilde = np.zeros(shape=(D.shape[0], D.shape[1], 3))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            memberships = np.array(
                [
                    fuzz.interp_membership(
                        ratings.universe, ratings[term].mf, 
                        D[i, j]
                    ) for term in ratings
                ]
            )
            a, b, c = function(memberships, ratings)
            D_tilde[i, j] = np.array([a, b, c])
    return D_tilde

ratings_terms = ['mr', 'r', 'm', 'b', 'mb']
ratings = ctrl.Antecedent(np.linspace(0, 10, 1000), 'avaliacao')
ratings.automf(number=7, names=ratings_terms, invert=False)

# from skfuzzy.control.visualization import FuzzyVariableVisualizer
# axs = None
# fig, axs = FuzzyVariableVisualizer(ratings).view(ax=axs)
# plt.show()

D = np.array(
    [
        [7, 7.5, 6, 8], 
        [8,   7.5, 7, 8], 
        [8, 2.5, 8, 9], 
        [6, 9.3, 2.5, 6], 
        [7,   9.4, 8, 2.5], 
        [9, 2.5, 8, 9], 
        [9, 0.2, 7, 9], 
        [2.3, 6.9, 7, 8], 
        [8, 0.5, 2.5, 9], 
    ]
)

# Define the decision matrix
D_tilde = make_decision_matrix(D, ratings, extract_nft_map)
# plot_fuzzy_set(ratings, ratings_terms[::-1], "avaliacao", "X", "μ(x)", D_tilde[:, :, 0].min(), D_tilde[:, :, 0].max(), 0, 1, "upper right")


# Define the weights
weights_terms = ['ni', 'pi', 'im', 'i', 'mi']
weights = ctrl.Antecedent(np.linspace(1, 5, 1000), 'peso')
weights.automf(number=7, names=weights_terms, invert=False)
# axs = None
# fig, axs = FuzzyVariableVisualizer(weights).view(ax=axs)
# plt.show()

W = np.array([3, 4, 2, 3])
W_tilde = make_weights_matrix(W, weights, extract_nft_map)
# plot_fuzzy_set(weights, weights_terms[::-1], "peso", "X", "μ(x)", W_tilde[:, 0].min(), W_tilde[:, 0].max(), 0, 1, "upper right")


criteria_types = [Criteria.BENEFIT, Criteria.BENEFIT, Criteria.BENEFIT, Criteria.BENEFIT]
ranking = fuzzy_topsis(D_tilde, W_tilde, criteria_types)
print(ranking)

