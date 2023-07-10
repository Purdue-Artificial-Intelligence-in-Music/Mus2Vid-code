python = "python3"


conda-save:
	@
conda-update:


emotion-data-process:
	$(python) -m src.emotion.data.process
emotion-data-util:
	$(python) -m src.emotion.data.util 
emotion-data: emotion-data-process emotion-data-util

emotion-features-best:
	$(python) -m src.emotion.features.best
emotion-features-extract:
	$(python) -m src.emotion.features.extract
emotion-features-util:
	$(python) -m src.emotion.features.util
emotion-features: emotion-features-best emotion-features-extract emotion-features-util

emotion-model-regressor:
	$(python) -m src.emotion.model.regressor
emotion-model-train:
	$(python) -m src.emotion.model.train
emotion-model-util:
	$(python) -m src.emotion.model.util
emotion-model: emotion-model-regressor emotion-model-train emotion-model-util

emotion-visualize-dataset:
	$(python) -m src.emotion.visualize.dataset
emotion-visualize-regression:
	$(python) -m src.emotion.visualize.regression
emotion-visualize-util:
	$(python) -m src.emotion.visualize.util
emotion-visualize: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util

emotion-main:
	$(python) -m src.emotion.main

emotion: emotion-data emotion-features emotion-model emotion-visualize emotion-main


# .PHONY targets don't result in an executable.
.PHONY: conda-save conda-update
.PHONY: emotion-data-process emotion-data-util emotion-data
.PHONY: emotion-features-best emotion-features-extract emotion-features-util emotion-features
.PHONY: emotion-model-regressor emotion-model-train emotion-model-util emotion-model
.PHONY: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util emotion-visualize
.PHONY: emotion-main emotion