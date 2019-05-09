.PHONY: all clean

all:
	pdflatex main
	sage main.sagetex.sage
	bibtex main
	pdflatex main
	pdflatex main

clean:  
	rm -f *.dvi *.aux *.sagetex.* *.ps *.pdf *.pyc *.log *.bbl *.blg \#*\# *~ 
