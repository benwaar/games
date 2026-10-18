.PHONY: check lint typecheck test invariants full

lint:
	ruff check src/ tests/

typecheck:
	mypy src/utala/

test:
	python run_tests.py -q

invariants:
	python -m tests.run_invariants --games 50

check: lint typecheck test

full: check invariants
