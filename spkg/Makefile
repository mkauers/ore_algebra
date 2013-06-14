
## boundle *.py files in .. into an spkg file

spkg:	
	rm -rf ore_algebra-0.1 .hg
	mkdir ore_algebra-0.1
	mkdir ore_algebra-0.1/src
	mkdir ore_algebra-0.1/src/ore_algebra
	cp ../*.py ore_algebra-0.1/src/ore_algebra
	echo "src/" > ore_algebra-0.1/.hgignore
	cp SPKG.txt ore_algebra-0.1
	cp spkg-install ore_algebra-0.1
	chmod u+x ore_algebra-0.1/spkg-install
	cp setup.py ore_algebra-0.1/src
	cd ore_algebra-0.1
	hg init
	cd .. 
	tar -jcf ore_algebra-0.1.spkg ore_algebra-0.1
	rm -rf ore_algebra-0.1

