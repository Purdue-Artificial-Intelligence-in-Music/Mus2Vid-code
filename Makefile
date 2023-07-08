python = "python3"


emotion-data-process:
	@cd src; $(python) -m emotion.data.process
emotion-data-util:
	@cd src; $(python) -m emotion.data.util 
emotion-data: emotion-data-process emotion-data-util

emotion-features-best:
	@cd src; $(python) -m emotion.features.best
emotion-features-extract:
	@cd src; $(python) -m emotion.features.extract
emotion-features-util:
	@cd src; $(python) -m emotion.features.util
emotion-features: emotion-features-best emotion-features-extract emotion-features-util

emotion-model-regressor:
	@cd src; $(python) -m emotion.model.regressor
emotion-model-train:
	@cd src; $(python) -m emotion.model.train
emotion-model-util:
	@cd src; $(python) -m emotion.model.util
emotion-model: emotion-model-regressor emotion-model-train emotion-model-util

emotion-visualize-dataset:
	@cd src; $(python) -m emotion.visualize.dataset
emotion-visualize-regression:
	@cd src; $(python) -m emotion.visualize.regression
emotion-visualize-util:
	@cd src; $(python) -m emotion.visualize.util
emotion-visualize: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util

emotion-main:
	@cd src; $(python) -m emotion.main

emotion: emotion-data emotion-features emotion-model emotion-visualize emotion-main


.PHONY: emotion-data-process emotion-data-util emotion-data
.PHONY: emotion-features-best emotion-features-extract emotion-features-util emotion-features
.PHONY: emotion-model-regressor emotion-model-train emotion-model-util emotion-model
.PHONY: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util emotion-visualize
.PHONY: emotion-main emotion