test:
	python3 test.py
.PHONY: test

errors:
	pylint -E $$(find . -name '*.py')
.PHONY: errors
