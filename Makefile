.PHONY: test

# Note: we are applying a 15s timeout so don't add any heavy tests here
test:
	timeout 15 python -m unittest discover -s src -p '*_test.py' -t src
