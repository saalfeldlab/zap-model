.PHONY: test

# Note: we are applying a 10s timeout so don't add any heavy tests here
test:
	timeout 10 python -m unittest discover -s src -p '*_test.py' -t src
