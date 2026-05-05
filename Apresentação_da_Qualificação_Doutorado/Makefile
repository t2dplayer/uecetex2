# Nome do arquivo principal (sem extensão)
MAIN = main

# Comandos
LATEXMK = latexmk
FLAGS = -pdf -interaction=nonstopmode -use-make

# Alvo principal
all: $(MAIN).pdf

# Compilação
$(MAIN).pdf: $(MAIN).tex setup.tex biblio.bib
	$(LATEXMK) $(FLAGS) $(MAIN).tex

# Assistir mudanças (compilação automática)
watch:
	$(LATEXMK) $(FLAGS) -pvc $(MAIN).tex

# Limpar arquivos auxiliares
clean:
	$(LATEXMK) -c
	rm -f *.nav *.snm *.vrb

# Limpar tudo, incluindo o PDF
distclean: clean
	$(LATEXMK) -C

# Abrir o PDF (usa o visualizador padrão do sistema)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

.PHONY: all clean distclean view watch
