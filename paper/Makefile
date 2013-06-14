
all:
	cp ../*.py ./ore_algebra
	pdflatex main
	sage main.sagetex.sage
	bibtex main
	pdflatex main

clean:  
	rm -f *.dvi *.aux *.sagetex.* *.ps *.pdf *.pyc *.log *.bbl *.blg \#*\# *~ 