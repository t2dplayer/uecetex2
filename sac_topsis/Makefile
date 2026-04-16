TEX=pdflatex
BIB=bibtex
OUTDIR=output

PROJECTS=main main-br

.PHONY: all clean

all: $(OUTDIR)/main.pdf $(OUTDIR)/main-br.pdf

$(OUTDIR):
	mkdir -p $(OUTDIR)

define build_pdf
mkdir -p $(OUTDIR)
$(TEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(1)
-$(BIB) $(OUTDIR)/$(1)
$(TEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(1)
$(TEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(1)
endef

$(OUTDIR)/main.pdf: main.tex | $(OUTDIR)
	$(call build_pdf,main)

$(OUTDIR)/main-br.pdf: main-br.tex | $(OUTDIR)
	$(call build_pdf,main-br)

clean:
	rm -rf $(OUTDIR)
