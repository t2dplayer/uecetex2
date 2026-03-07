# Makefile for compiling LaTeX documents

# Default settings
LATEX = pdflatex
BIBTEX = bibtex
MAIN_DOC = sn-article
DIFF_DOC = sn-article-diff

# Default target: compile the main document
all: $(MAIN_DOC).pdf

# Compile the main document
$(MAIN_DOC).pdf: $(MAIN_DOC).tex
	$(LATEX) $(MAIN_DOC)
	$(BIBTEX) $(MAIN_DOC)
	$(LATEX) $(MAIN_DOC)
	$(LATEX) $(MAIN_DOC)

# Target for compiling the diff version
highlights: $(DIFF_DOC).pdf

# Compile the diff document
$(DIFF_DOC).pdf: $(DIFF_DOC).tex
	$(LATEX) $(DIFF_DOC)
	$(BIBTEX) $(DIFF_DOC)
	$(LATEX) $(DIFF_DOC)
	$(LATEX) $(DIFF_DOC)

# Clean up temporary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean all generated files including PDFs
cleanall: clean
	rm -f $(MAIN_DOC).pdf $(DIFF_DOC).pdf

.PHONY: all highlights clean cleanall