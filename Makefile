test:
	python3 test.py
.PHONY: test

errors:
	pylint -E $$(find . -name '*.py') --rcfile=./.pylintrc
.PHONY: errors

mypy:
	MYPYPATH=$$HOME/ATB_ONE mypy --fast-parser $$(find . -name '*.py' |  sed "s|^\./||" )
.PHONY:
