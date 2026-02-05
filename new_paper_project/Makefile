PROJECT=main
TEX=pdflatex
BIB=bibtex
BUILDTEX=$(TEX) -interaction=nonstopmode $(PROJECT)
BUILDBIB=$(BIB) $(PROJECT)

.PHONY: all clean

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex
	$(BUILDTEX)
	-$(BUILDBIB)
	$(BUILDTEX)
	$(BUILDTEX)

clean:
	rm -f *.aux *.log *.out *.dvi *.bbl *.blg *.toc *.lof *.lot $(PROJECT).pdf
