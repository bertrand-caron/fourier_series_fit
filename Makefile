PYTHONPATH = PYTHONPATH=$(shell dirname $$PWD)
PYHON_EXEC = $(PYTHONPATH) python3

test: install
	$(PYHON_EXEC) test.py
.PHONY: test

errors:
	pylint -E $$(find . -name '*.py') --rcfile=./.pylintrc
.PHONY: errors

mypy:
	MYPYPATH=$$HOME/ATB_ONE mypy --fast-parser $$(find . -name '*.py' |  sed "s|^\./||" )
.PHONY:

install: requirements.txt
	pip3 install -r $<
.PHONY: install
