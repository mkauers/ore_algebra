
## bundle files in ../src (except for shift.spyx) into an spkg file

spkg:	
	rm -rf ore_algebra-0.2 .hg
	# copy tracked files only (and copy the commited version rather than
	# the one in the source tree)
	(cd .. && git archive --format=tar --prefix=ore_algebra-0.2/ HEAD src) \
	    | tar x
	rm ore_algebra-0.2/src/ore_algebra/shift.spyx
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

