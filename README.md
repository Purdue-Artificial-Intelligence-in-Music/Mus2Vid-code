# Mus2Vid

Welcome to the repository for the Purdue research project Mus2Vid:
Automatic generation of video accompaniment for classical music.
This project is affiliated with the CAM2 research lab, and is under the
direction of Professor Yung-Hsiang Lu (ECE) and Professor Kristen Yeon-Ji Yun (Music).

## Table of Contents

- [Setup](#setup)
    - [Clone the repository](#clone-the-repository)
    - [Conda environment](#conda-environment)
    - [Download raw data](#download-raw-data)
- [Documentation](#documentation)
    - [Reference docs](#reference-docs)
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
| [Miniconda or Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) | Package manager |
| Linux or Windows Subsystem for Linux (WSL) | Something about linux |
| deam_dataset.zip | See [Download Raw Data](#download-raw-data) |

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

Docstrings are used for commenting packages, modules, classes,
functions, and methods. Inline comments just use regular python comments
and should be used sparingly.

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