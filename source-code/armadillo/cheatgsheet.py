from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 8, 'Armadillo C++ Library - THE COMPLETE REFERENCE', 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 6, 'Todas as funcoes, classes e utilitarios (Doc v12+)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Pagina ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, title):
        self.ln(4)
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(50, 50, 100) # Azul escuro
        self.set_text_color(255, 255, 255) # Texto branco
        self.cell(0, 7, " " + title, 0, 1, 'L', 1)
        self.ln(2)

    def command_block(self, syntax, desc, example):
        # Layout compacto mas legivel
        self.set_font('Courier', 'B', 9) 
        self.set_text_color(0, 0, 150) # Azul
        
        # SINTAXE
        self.cell(0, 4, syntax, 0, 1)
        
        # DESCRICAO
        self.set_font('Arial', '', 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4, "  " + desc)
        
        # EXEMPLO
        if example:
            self.set_font('Courier', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 4, "  >> " + example, 0, 1)
        
        self.ln(2)

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

data = {
    "1. ESTRUTURAS DE DADOS (CORE)": [
        ("Mat<type> / mat / fmat / cx_mat", "Matrizes densas (Double, Float, Complex).", "mat A(10,10); cx_mat C;"),
        ("Col<type> / vec / uvec / ivec", "Vetores Coluna (Double, Unsigned Int, Int).", "vec v(100); uvec indices;"),
        ("Row<type> / rowvec", "Vetores Linha.", "rowvec r = v.t();"),
        ("Cube<type> / cube", "Cubo 3D (Linhas, Colunas, Fatias).", "cube Q(10,10,5);"),
        ("SpMat<type> / sp_mat", "Matriz Esparsa (Formato CSC).", "sp_mat S(1000,1000);"),
        ("field<type>", "Matriz de objetos arbitrários.", "field<mat> F(2,2); F(0,0) = A;")
    ],
    "2. INICIALIZACAO E GERADORES": [
        ("zeros(l,c) / ones(l,c)", "Preenche com 0s ou 1s.", "mat A = zeros(5,5);"),
        ("eye(l,c)", "Identidade (Diagonal=1).", "mat I = eye(5,5);"),
        ("randu(l,c) / randn(l,c)", "Uniforme [0,1] e Normal (0,1).", "mat R = randn(10,10);"),
        ("randi(l,c, distr_param(a,b))", "Inteiros aleatorios [a,b].", "imat I = randi(5,5, distr_param(0,10));"),
        ("randg(l,c, distr_param(a,b))", "Distribuicao Gamma.", "mat G = randg(10,10, distr_param(2,1));"),
        ("linspace(start, end, N)", "Vetor linear (inclui extremos).", "vec v = linspace(0,1,100);"),
        ("logspace(start, end, N)", "Vetor logaritmico.", "vec v = logspace(0,5,100);"),
        ("regspace(start, delta, end)", "Vetor com passo delta.", "vec v = regspace(0, 2, 10);"),
        ("toeplitz(v)", "Matriz Toeplitz.", "mat T = toeplitz(v);"),
        ("circ_toeplitz(v)", "Matriz Toeplitz Circular.", "mat C = circ_toeplitz(v);"),
        ("operator <<", "Inicializacao por lista.", "mat A = { {1,2}, {3,4} };")
    ],
    "3. ACESSO, SUBMATRIZES E ITERADORES": [
        ("A(i,j) / A(k)", "Acesso (sem bounds check).", "double x = A(5,5);"),
        ("A.at(i,j)", "Acesso seguro (com bounds check).", "try { val = A.at(99,99); }"),
        ("A.row(i) / A.col(j)", "Acessa Linha/Coluna especifica.", "A.row(0) = randu<rowvec>(10);"),
        ("A.rows(a,b) / A.cols(a,b)", "Intervalo continuo de linhas/cols.", "mat B = A.cols(0, 4);"),
        ("A.submat(r1,c1, r2,c2)", "Submatriz (indices inclusivos).", "mat S = A.submat(0,0, 4,4);"),
        ("A.submat(vector_idx)", "Submatriz via indices nao-contiguos.", "mat S = A.submat(uvec_rows, uvec_cols);"),
        ("A.head_rows(k) / A.tail_rows(k)", "Topo e Fundo.", "mat H = A.head_rows(5);"),
        ("A.diag(k)", "Diagonal (k=0 principal).", "vec d = A.diag();"),
        ("A.begin() / A.end()", "Iteradores STL (para loops/algoritmos).", "for(double& val : A) { val *= 2; }")
    ],
    "4. PROPRIEDADES E INFORMACOES": [
        ("n_rows / n_cols / n_elem", "Dimensoes (variaveis membro).", "uword n = A.n_rows;"),
        ("size()", "Retorna tamanho.", "cout << A.size() << endl;"),
        ("is_empty()", "Verifica se vazio.", "if(A.is_empty()) ..."),
        ("is_square()", "Verifica se L == C.", "bool b = A.is_square();"),
        ("is_symmetric()", "Verifica simetria.", "bool b = A.is_symmetric();"),
        ("is_finite() / has_nan() / has_inf()", "Checagem de integridade numerica.", "if(A.has_nan()) A.replace(datum::nan, 0);")
    ],
    "5. OPERACOES ELEMENT-WISE (MATEMATICA)": [
        ("A + B / A - B", "Soma e Subtracao.", "mat C = A + B;"),
        ("A % B", "Produto SCHUR (Elemento-a-Elemento).", "mat C = A % B;"),
        ("A / B", "Divisao Elemento-a-Elemento.", "mat C = A / B;"),
        ("abs(A) / sqrt(A) / square(A)", "Modulo, Raiz, Quadrado.", "mat S = sqrt(A);"),
        ("exp(A) / log(A) / log10(A)", "Exponencial e Logaritmos.", "mat E = exp(A);"),
        ("sin / cos / tan / cot", "Trigonometria.", "mat S = sin(A);"),
        ("asin / acos / atan / atan2", "Trigonometria Inversa.", "mat AS = asin(A);"),
        ("sinh / cosh / tanh", "Hiperbolicas.", "mat T = tanh(A);"),
        ("round / floor / ceil / trunc", "Arredondamentos.", "mat R = round(A);"),
        ("sign(A)", "Retorna -1, 0, ou 1.", "mat S = sign(A);"),
        ("clamp(A, min, max)", "Limita valores (Clip/ReLU).", "mat C = clamp(A, 0.0, 1.0);")
    ],
    "6. ALGEBRA LINEAR (SOLVERS & DECOMPOSICOES)": [
        ("solve(A, B)", "Resolve AX = B.", "vec x = solve(A, B);"),
        ("inv(A) / inv_sympd(A)", "Inversa (Geral e Simetrica Pos-Def).", "mat Ai = inv(A);"),
        ("pinv(A)", "Pseudo-Inversa (SVD based).", "mat P = pinv(A);"),
        ("det(A) / log_det(A)", "Determinante.", "double d = det(A);"),
        ("rank(A)", "Posto numerico.", "uword r = rank(A);"),
        ("trace(A)", "Traco (soma diagonal).", "double t = trace(A);"),
        ("norm(A, p)", "Norma (1, 2, fro, inf).", "double n = norm(A, \"fro\");"),
        ("cond(A) / rcond(A)", "Numero de Condicionamento.", "double c = cond(A);"),
        ("null(A)", "Base do Espaco Nulo.", "mat N = null(A);"),
        ("orth(A)", "Base Ortonormal.", "mat O = orth(A);"),
        ("eig_sym(val, vec, A)", "Autovalores (Simetrica).", "eig_sym(evals, evecs, A);"),
        ("eig_gen(val, vec, A)", "Autovalores (Geral).", "eig_gen(evals, evecs, A);"),
        ("eig_pair(val, vec, A, B)", "Autovalores Generalizados (A, B).", "eig_pair(val, vec, A, B);"),
        ("svd(U, s, V, A)", "Decomposicao de Valor Singular.", "svd(U, s, V, A);"),
        ("svd_econ(U, s, V, A)", "SVD Economico (menos memoria).", "svd_econ(U, s, V, A);"),
        ("chol(A)", "Cholesky (A = R.t() * R).", "mat R = chol(A);"),
        ("qr(Q, R, A)", "QR Decomposicao.", "qr(Q, R, A);"),
        ("lu(L, U, P, A)", "LU Decomposicao (com Pivot).", "lu(L, U, P, A);"),
        ("schur(U, S, A)", "Decomposicao de Schur.", "schur(U, S, A);"),
        ("hess(H, A)", "Decomposicao de Hessenberg.", "hess(H, A);"),
        ("sylvester(A, B, C)", "Resolve eq. Sylvester AX + XB = C.", "mat X = sylvester(A, B, C);")
    ],
    "7. ESTATISTICA E AGREGACAO": [
        ("mean(A, dim) / median(A, dim)", "Media e Mediana (0=col, 1=row).", "vec m = mean(A, 0);"),
        ("var(A) / stddev(A)", "Variancia e Desvio Padrao.", "double v = var(v);"),
        ("min(A) / max(A)", "Valores Extremos.", "double mx = A.max();"),
        ("index_min(A) / index_max(A)", "Indices dos Extremos.", "uword i = A.index_max();"),
        ("range(A)", "Diferenca Max - Min.", "vec r = range(A);"),
        ("cov(A) / cor(A)", "Matriz de Covariancia/Correlacao.", "mat C = cov(A);"),
        ("hist(v, centers) / histc(v, edges)", "Histogramas.", "uvec counts = hist(v, 10);"),
        ("quantile(A, P)", "Quantis (P vector [0,1]).", "vec q = quantile(A, vec{0.25, 0.75});")
    ],
    "8. MANIPULACAO DE DADOS E ESTRUTURA": [
        ("join_rows(A, B) / join_cols(A, B)", "Concatena matrizes.", "mat C = join_rows(A, B);"),
        ("reshape(l, c)", "Altera dimensoes mantendo dados.", "A.reshape(10, 2);"),
        ("resize(l, c)", "Altera dimensoes (pode perder dados).", "A.resize(100, 100);"),
        ("fliplr(A) / flipud(A)", "Espelha matriz.", "mat B = fliplr(A);"),
        ("shift(A, n) / circshift(A, n)", "Deslocamento de elementos.", "mat S = shift(A, -1);"),
        ("shuffle(A)", "Embaralha ordem.", "vec v = shuffle(v);"),
        ("sort(v) / sort_index(v)", "Ordena e retorna indices.", "uvec idx = sort_index(v);"),
        ("unique(A)", "Valores unicos.", "vec u = unique(A);"),
        ("vectorise(A)", "Converte matriz em vetor.", "vec v = vectorise(A);"),
        ("ind2sub(size, idx) / sub2ind", "Converte indices lineares/matriciais.", "uvec sub = ind2sub(size(A), idx);")
    ],
    "9. PROCESSAMENTO DE SINAL E IMAGEM": [
        ("fft(v) / ifft(v)", "Fast Fourier Transform (1D).", "cx_vec f = fft(v);"),
        ("fft2(A) / ifft2(A)", "FFT 2D (Matrizes).", "cx_mat f = fft2(A);"),
        ("conv(u, v, shape)", "Convolucao 1D.", "vec c = conv(u, v, \"same\");"),
        ("conv2(A, B, shape)", "Convolucao 2D.", "mat c = conv2(img, kernel, \"same\");"),
        ("interp1(x, y, xi)", "Interpolacao 1D.", "vec yi = interp1(x, y, xi, \"linear\");")
    ],
    "10. MATRIZES ESPARSAS (SPARSE)": [
        ("speye / spones / sprandu", "Geradores Esparsos.", "sp_mat S = sprandu(1000, 1000, 0.01);"),
        ("spsolve(A, B)", "Solver Linear para Esparsas.", "vec x = spsolve(SpA, B);"),
        ("eigs_sym / eigs_gen", "Autovalores Esparsos (ARPACK).", "eigs_sym(val, vec, SpA, 5, \"lm\");"),
        ("svds(SpA, k)", "SVD Truncado (Esparso).", "svds(U, s, V, SpA, 5);")
    ],
    "11. MACHINE LEARNING & ESTATISTICA AVANCADA": [
        ("kmeans", "Clustering (Lloyd's algo).", "kmeans(means, data, k, random_subset, 10, true);"),
        ("gmm_diag / gmm_full", "Gaussian Mixture Models.", "gmm.learn(data, k, kmeans_mode);"),
        ("princomp", "Principal Component Analysis.", "princomp(coeff, score, latent, A);"),
        ("mvnrnd", "Normal Multivariada Random.", "mat X = mvnrnd(mu, sigma, N);"),
        ("chi2rnd / wishrnd", "Chi-Square e Wishart Random.", "mat W = wishrnd(Sigma, df);"),
        ("normpdf / log_normpdf", "Prob Density Function.", "vec p = normpdf(x, mu, sigma);")
    ],
    "12. LAMBDAS, FUNCTORS E C++11": [
        ("transform(lambda)", "Transforma inplace.", "A.transform( [](double x){ return x>0?x:0; } );"),
        ("for_each(lambda)", "Itera (const ou side-effect).", "A.for_each( [](double x){ cout << x; } );"),
        ("imbue(lambda)", "Preenche com lambda.", "A.imbue( [](){ return rand()%10; } );"),
        ("find(condicao)", "Retorna indices.", "uvec idx = find(A > 0.5);"),
        ("elem(idx)", "Acessa via indices.", "A.elem(idx).zeros();"),
        ("replace(old, new)", "Substituicao direta.", "A.replace(datum::nan, 0);"),
        ("clean(threshold)", "Zera ruidos (numeros muito pequenos).", "A.clean(datum::eps);"),
        ("any() / all()", "Logica booleana.", "if(any(vector > 10)) ...")
    ],
    "13. NUMEROS COMPLEXOS (CX_MAT)": [
        ("real(C) / imag(C)", "Extrai partes real/imag.", "mat R = real(C);"),
        ("conj(C)", "Conjugado.", "cx_mat C2 = conj(C);"),
        ("abs(C) / arg(C)", "Magnitude e Fase (Angulo).", "mat M = abs(C);")
    ],
    "14. CONSTANTES E IO": [
        ("datum::pi, inf, nan, eps", "Constantes.", "double p = datum::pi;"),
        ("save(nome, formato)", "Salva Arquivo (auto-detect).", "A.save(\"data.bin\", arma_binary);"),
        ("load(nome)", "Carrega Arquivo.", "A.load(\"data.csv\");"),
        ("print(msg)", "Imprime no stdout.", "A.print(\"Matrix A:\");"),
        ("raw_print(msg)", "Imprime sem formatacao.", "A.raw_print();")
    ]
}

# Gerar o conteudo
for section, items in data.items():
    pdf.chapter_title(section)
    for syntax, desc, example in items:
        pdf.command_block(syntax, desc, example)

# Salvar
filename = 'armadillo_cheatsheet_bible.pdf'
pdf.output(filename, 'F')
print(f"PDF gerado com sucesso: {filename}")