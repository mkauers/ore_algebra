.PHONY: build install test testlong

build:
	sage -python setup.py build_ext --inplace

install:
	sage -pip install --upgrade --no-index .

test: build
	PYTHONPATH=src sage -tp --force-lib src

testlong: build
	PYTHONPATH=src sage -tp --long --warn-long 2 --force-lib src
