# Makefile for LaTeX project "A Context-Oriented Framework for Computation Offloading"

# Main document name (without extension)
MAIN = main

# LaTeX compiler (using LuaLaTeX for fontspec support)
LATEX = lualatex

# Bibliography compiler
BIBTEX = bibtex

# Additional files
TEX_FILES = content.tex editor.tex reviewers.tex rev1.tex rev2.tex rev3.tex rev4.tex

# Output directory
OUTPUT_DIR = output

# Default target
all: $(MAIN).pdf

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Compile the main document
$(MAIN).pdf: $(MAIN).tex $(TEX_FILES) | $(OUTPUT_DIR)
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN)
	@if grep -q "\\\\cite" $(MAIN).tex $(TEX_FILES); then \
		cd $(OUTPUT_DIR) && $(BIBTEX) $(MAIN); \
		cd .. && $(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN); \
	fi
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN)
	cp $(OUTPUT_DIR)/$(MAIN).pdf .

# Clean auxiliary files
clean:
	rm -f $(OUTPUT_DIR)/*.aux $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.out \
	      $(OUTPUT_DIR)/*.toc $(OUTPUT_DIR)/*.bbl $(OUTPUT_DIR)/*.blg \
	      $(OUTPUT_DIR)/*.lof $(OUTPUT_DIR)/*.lot $(OUTPUT_DIR)/*.fdb_latexmk \
	      $(OUTPUT_DIR)/*.fls $(OUTPUT_DIR)/*.synctex.gz

# Clean all generated files
distclean: clean
	rm -f $(MAIN).pdf $(OUTPUT_DIR)/$(MAIN).pdf
	rmdir $(OUTPUT_DIR) 2>/dev/null || true

# Watch for changes and recompile automatically (requires inotifywait)
watch:
	@echo "Watching for changes in *.tex files..."
	@while true; do \
		inotifywait -e modify -e create -e delete -e move *.tex; \
		$(MAKE) all; \
		echo "Recompiled at $$(date)"; \
	done

# Target to explicitly create PDF with highlights
highlights: $(MAIN).tex $(TEX_FILES) | $(OUTPUT_DIR)
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN)
	@if grep -q "\\\\cite" $(MAIN).tex $(TEX_FILES); then \
		cd $(OUTPUT_DIR) && $(BIBTEX) $(MAIN); \
		cd .. && $(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN); \
	fi
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTPUT_DIR) $(MAIN)
	cp $(OUTPUT_DIR)/$(MAIN).pdf ./$(MAIN)_with_highlights.pdf

.PHONY: all clean distclean watch highlights