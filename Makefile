# Makefile for ML Training + Flask App

install:
	pip install -r requirements.txt

train:
	python src/models/train_hardness.py
	python src/models/train_oxidation.py

evaluate:
	python src/models/evaluate.py

run:
    python -m src.app.app


clean:
	rm -rf __pycache__
	find . -name '*.pyc' -delete
