
## boundle *.py files in .. into an spkg file

spkg:	
	rm -rf ore_algebra-0.2 .hg
	mkdir ore_algebra-0.2
	mkdir ore_algebra-0.2/src
	mkdir ore_algebra-0.2/src/ore_algebra
	cp ../*.py ore_algebra-0.2/src/ore_algebra
	echo "src/" > ore_algebra-0.2/.hgignore
	cp SPKG.txt ore_algebra-0.2
	cp spkg-install ore_algebra-0.2
	chmod u+x ore_algebra-0.2/spkg-install
	cp setup.py ore_algebra-0.2/src
	cd ore_algebra-0.2
	hg init
	cd .. 
	tar -jcf ore_algebra-0.2.spkg ore_algebra-0.2
	rm -rf ore_algebra-0.2

