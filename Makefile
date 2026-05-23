sentence-tokenize:
	python -m basemodel.src.data.sentence_tokenize

train-tokenizer:
	python -m basemodel.src.training.train_tokenizer

processed-datasets:
	python -m basemodel.src.data.processed --train-split true --corpus false --train-ratio 0.95

validate-tokenizer:
	python -m basemodel.tests.validate_tokenizer