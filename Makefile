PYTHONPATH ?= .

.PHONY: install-linters install lint test-distributed

install-linters:
	pip install pre-commit
	pre-commit install

install:
	pip install --no-build-isolation -f https://whl.natten.org -r requirements.txt

lint:
	python3 ci/check_licenses.py
	pre-commit run -a

test-distributed:
	PYTHONPATH=$(PYTHONPATH) torchrun --master_port $${MASTER_PORT:-29500} --local_ranks_filter 0 --nproc_per_node 8 -m pytest tests/distributed -v
