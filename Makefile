.PHONY: test

# Run unit tests
test:
	python -m unittest discover -s src -p '*_test.py' -t src
