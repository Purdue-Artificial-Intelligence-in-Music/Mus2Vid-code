# Mus2Vid: Automatic generation of video accompaniment for classical music

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

Welcome to the repository for the Purdue research project Mus2Vid:
Automatic generation of video accompaniment for classical music.
This project is affiliated with the
[CAM<sup>2</sup> research lab <i class="fa fa-external-link" style="font-size:12px"></i>](https://engineering.purdue.edu/VIP/teams/cam2)
and is under the direction of
Professor Yung-Hsiang Lu (ECE) and
Professor Kristen Yeon-Ji Yun (Music).

## Table of Contents

- [Setup](#setup)
    - [Clone the repository](#clone-the-repository)
    - [Conda environment](#conda-environment)
    - [Download raw data](#download-raw-data)
- [Documentation](#documentation)
    - [Reference docs](#reference-docs)
        - [Docstrings](#docstrings)
    - [Process docs](#process-docs)
- [Repository Structure](#repository-structure-current)
    - [Current](#current)
    - [Proposed](#proposed)

# Setup

### Clone the repository

    $ git clone git@github.com:Mus2Vid/Mus2Vid-code.git

### Requirements

| Name | Description |
| ---- | ----------- |
| [Miniconda <i class="fa fa-external-link" style="font-size:12px"></i>](https://docs.conda.io/en/main/miniconda.html) or [Conda <i class="fa fa-external-link" style="font-size:12px"></i>](https://www.anaconda.com/download/) | Package manager |
| Linux or Windows Subsystem for Linux (WSL) | Something about linux |
| `deam_dataset.zip` | See [Download Raw Data](#download-raw-data) |

### Conda environment

Create or update the conda environment by running this command from
the root of the repository 

    $ make conda

Save conda environment to `environment.yml`

    $ make conda-save

*Note: these commands work regardless of whether or not
the conda environment is currently activated*.

### Download raw data

Download `deam_dataset.zip`, and extract its contents into `data/raw`.

Your directory should look like this after extracting `deam_dataset.zip`:

```
Mus2Vid-code/
├── ...
├── data/
│   └── raw/
│       └── deam_dataset/
│           ├── DEAM_Annotations/
│           │   └── ...
│           ├── DEAM_audio/
│           │   └── ...
│           ├── features/
│           │   └── ...
│           └── metadata/
│               └── ...
└── ...
```

# Documentation

### Reference docs

Docstrings are used for code comments for packages, modules, classes,
functions, and methods. Inline comments just use regular python comments
and should be used *sparingly*.

As a rule of thumb, code comments should not be used to convey info that
can be easily gathered from the code itself.

If short on time, focus on writing comments for broad things (e.g.,
modules, classes) and/or segments of code that might be harder to
understand (e.g., nested list comprehensions, complex algorithm).

#### Docstrings

*Function*

Functions should generally have a section for Parameters and Returns.
Returns should have type hints and&mdash;if necessary&mdash;a short description.

```py
def get_va_values(audio_filepath: str) -> tuple[float, float]:
    """Process audio at given filepath and return valence and arousal values.

    Parameters
    ----------
    audio_filepath
        Filepath relative to repository root.

    Returns
    -------
    valence: float
        A float between 1 and 9.
    arousal: float
        A float between 1 and 9.
    """
    valence_regressor = EmotionRegressor()
    arousal_regressor = EmotionRegressor()
    valence_regressor.load("valence_regressor")
    arousal_regressor.load("arousal_regressor")

    opensmile_features = extract_opensmile_features([audio_filepath])
    opensmile_valence_features, opensmile_arousal_features = get_best_opensmile_features(opensmile_features)

    valence = valence_regressor.predict(opensmile_valence_features)[0]
    arousal = arousal_regressor.predict(opensmile_arousal_features)[0]

    return valence, arousal
```

```py
def get_valence_targets() -> pd.Series:
    """Return a pandas.Series of target values for valence."""
    return pd.read_csv(ANNOTATIONS_PATH)["valence_mean"]
```

*Classes and methods*

Classes with attributes will have an Attributes section.
The methods should not be included in the class's docstring.
Instead, each method should have its own docstring.

```py
class EmotionRegressor():
    """A Support Vector Regressor (SVR) for predicting valence and
    arousal values given the pre-extracted features of an audio file.
    
    EmotionRegressor can be used to predict both valence and
    arousal values from an audio file. Separate instances of EmotionRegressor
    should be used for valence and arousal.
    """

    def __init__(self, epsilon=0.2) -> None:
        """Initialize regressor with option to set epsilon value."""
        self.svr = SVR(epsilon=epsilon)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        """Fit model to provided set of features and targets."""
        self.svr.fit(features, targets)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict and return valence and arousal values."""
        return self.svr.predict(inputs)

    def save(self, filename: str) -> None:
        """Save the current model to a file."""
        if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
        joblib.dump(self.svr, f"{MODEL_DIR}/{filename}.{MODEL_EXT}")

    def load(self, filename: str) -> None:
        """Load a model from a file."""
        self.svr = joblib.load(f"{MODEL_DIR}/{filename}.{MODEL_EXT}") 
```

### Process docs

Google Drive will be used to host our design docs.

# Repository Structure 

### Current

```
Mus2Vid-code/
├── environment.yml
├── Makefile
├── README.md
├── Basic_Pitch/
├── Emotion_NN/
├── Genre_NN/
│   ├── chord_detection/
│   └── pkls/
├── Image_Generation/
├── Multithreading work/
├── Prompt/
├── Prototype/
│   ├── test_mp3/
│   └── utils/
├── Random junk/
├── basic-pitch-modified/
├── data/
│   ├── interim/
│   │   └── features/
│   ├── processed/
│   │   ├── annotations/
│   │   ├── audio/
│   │   ├── features/
│   │   └── targets/
│   └── raw/
│       └── deam_dataset/
│           ├── DEAM_Annotations/
│           │   └── annotations/
│           │       ├── annotations averaged per song/
│           │       │   ├── dynamic (per second annotations)/
│           │       │   └── song_level/
│           │       └── annotations per each rater/
│           │           ├── dynamic (per second annotations)/
│           │           │   ├── arousal/
│           │           │   └── valence/
│           │           └── song_level/
│           ├── DEAM_audio/
│           │   └── MEMD_audio/
│           ├── features/
│           │   └── features/
│           └── metadata/
│               └── metadata/
├── models/
│   └── emotion/
├── reports/
│   └── figures/
└── src/
    ├── __init__.py
    └── emotion/
        ├── __init__.py
        ├── main.py
        ├── data/
        │   ├── __init__.py
        │   ├── process.py
        │   └── util.py
        ├── features/
        │   ├── __init__.py
        │   ├── best.py
        │   ├── extract.py
        │   └── util.py
        ├── model/
        │   ├── __init__.py
        │   ├── regressor.py
        │   ├── train.py
        │   └── util.py
        └── visualize/
            ├── __init__.py
            ├── dataset.py
            ├── regression.py
            └── util.py
```

### Proposed

```
Mus2Vid-code/
├── LICENSE
├── Makefile
├── README.md
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
│   └── emotion/
├── notebooks/
│   ├── basic_pitch.ipynb
│   ├── demo_for_meeting.ipynb
│   ├── multithreading_stuff.ipynb
│   ├── prototype.ipynb
│   └── some_stuff.ipynb
├── reports/
│   └── figures/
└── src/
    ├── __init__.py
    ├── main.py
    ├── emotion/
    │   ├── __init__.py
    |   └── ...
    ├── genre/
    │   ├── __init__.py
    |   └── ...
    ├── image_gen/
    │   ├── __init__.py
    |   └── ...
    └── prompt_gen/
        ├── __init__.py
        └── ...
```