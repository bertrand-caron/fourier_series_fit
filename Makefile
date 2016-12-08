test:
	python3 test.py
.PHONY: test

errors:
	pylint -E $$(find . -name '*.py')
.PHONY: errors

mypy:
	MYPYPATH=$$HOME/ATB_ONE mypy $$(find . -name '*.py' |  sed "s|^\./||" )
.PHONY:
