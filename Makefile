emotion_data = "./src/emotion/data"
emotion_features = "./src/emotion/features"
emotion_model = "./src/emotion/model"
emotion_visualize = "./src/emotion/visualize"
python = "python3"


emotion-data-process:
	@cd $(emotion_data); $(python) -m process
emotion-data-util:
	@cd $(emotion_data); $(python) -m util 
emotion-data: emotion-data-process emotion-data-util

emotion-features-best:
	@cd $(emotion_features); $(python) -m best
emotion-features-extract:
	@cd $(emotion_features); $(python) -m extract
emotion-features-util:
	@cd $(emotion_features); $(python) -m util
emotion-features: emotion-features-best emotion-features-extract emotion-features-util

emotion-model-regressor:
	@cd $(emotion_model); $(python) -m regressor
emotion-model-train:
	@cd $(emotion_model); $(python) -m train
emotion-model-util:
	@cd $(emotion_model); $(python) -m util
emotion-model: emotion-model-regressor emotion-model-train emotion-model-util

emotion-visualize-dataset:
	@cd $(emotion_visualize); $(python) -m dataset
emotion-visualize-regression:
	@cd $(emotion_visualize); $(python) -m regression
emotion-visualize-util:
	@cd $(emotion_visualize); $(python) -m util
emotion-visualize: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util

emotion: emotion-data emotion-features emotion-model emotion-visualize


.PHONY: emotion-data-process emotion-data-util emotion-data
.PHONY: emotion-features-best emotion-features-extract emotion-features-util emotion-features
.PHONY: emotion-model-regressor emotion-model-train emotion-model-util emotion-model
.PHONY: emotion-visualize-dataset emotion-visualize-regression emotion-visualize-util emotion-visualize
.PHONY: emotion