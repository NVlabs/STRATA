PYTHONPATH ?= .

.PHONY: install-linters install lint test-distributed

install-linters:
	pip install pre-commit
	pre-commit install

install:
	pip install --no-build-isolation -r requirements.txt
	pip install -f https://whl.natten.org "natten>=0.21.5"
	pip install --no-deps --no-build-isolation https://github.com/NVlabs/earth2grid/archive/3415382a37e414a867d4c6ae2519e3e8afdbece1.tar.gz
	pip install "warp-lang>=1.14" jaxtyping
	pip install --no-deps https://github.com/NVIDIA/physicsnemo/archive/07cdcc8b65f10bee733a32a77b4b76e20c70c54c.tar.gz

lint:
	python3 ci/check_licenses.py
	pre-commit run -a

test-distributed:
	PYTHONPATH=$(PYTHONPATH) torchrun --master_port $${MASTER_PORT:-29500} --local_ranks_filter 0 --nproc_per_node 8 -m pytest tests/distributed -v
