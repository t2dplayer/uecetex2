# Nome do projeto
PROJECT = documento

# Comandos
LATEXMK = latexmk
FLAGS   = -pdf -synctex=1 -interaction=nonstopmode -file-line-error

# --- Regras ---

.PHONY: all clean

all: $(PROJECT).pdf

# O segredo está aqui: adicionamos .PHONY ao target do PDF indiretamente
# Isso força o 'make' a executar o latexmk toda vez.
# O latexmk, por sua vez, é inteligente e só vai compilar se houver mudanças reais.
$(PROJECT).pdf: $(PROJECT).tex FORCE_MAKE
	@echo "Verificando alterações..."
	$(LATEXMK) $(FLAGS) $(PROJECT).tex

# Regra vazia para forçar a execução
FORCE_MAKE:

clean:
	$(LATEXMK) -c

clean-all:
	$(LATEXMK) -C