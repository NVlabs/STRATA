PYTHONPATH ?= .

.PHONY: install-linters install lint test-distributed

install-linters:
	pip install pre-commit
	pre-commit install

install:
	pip install --no-build-isolation -r requirements.txt
	pip install -f https://whl.natten.org "natten>=0.21.7"
	pip install --no-deps --no-build-isolation https://github.com/NVlabs/earth2grid/archive/6e39b069f0558cd0eff0ec666ede5e125d79b4ff.tar.gz
	pip install "warp-lang>=1.17" jaxtyping "GitPython>=3.1.62" "hydra-core>=1.3.7" "termcolor>=3.3.0" "treelib>=1.8.0"
	pip install --no-deps https://github.com/NVIDIA/physicsnemo/archive/fae6bb2b0e96488a052d9627d41e6dc94653f06a.tar.gz

lint:
	python3 ci/check_licenses.py
	pre-commit run -a

test-distributed:
	PYTHONPATH=$(PYTHONPATH) torchrun --master_port $${MASTER_PORT:-29500} --local_ranks_filter 0 --nproc_per_node 8 -m pytest tests/distributed -v
