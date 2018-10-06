# Activity Recognition based on features of objects detected

In this project, I use a dataset of features of objects detected in Activity Net's Exterior Maintenance Set of activities. The features are inferred using MRCNN e.g. confidence levels, bounding boxes, some feature of the masks. I use the dataset to train a standard LSTM model to classify the videos of the activities correctly. 

## Getting Started

You need to clone the reporistory to your local machine. Run this command in your terminal where you like to clone the project

```
git clone https://github.com/melbrbry/Activity-Recognition
```

### Prerequisites

Required packages (use pip install on linux):  
numpy  
tensorflow  
matplotlib

## Repository Description
The repository has 4 branches + master branch organized as:
  - binary branch: contains the version of the project where only binary features on whether an object is detected or not are used to train the LSTM.
  - confidence branch: the confidence percentage of the objects detected are used instead of only binary features.
  - bounding-box branch: plus the confidence the bounding box features are also used.
  - mask branch: plus the confidence and the ounding box features also some of mask features e.g. mask area and centroid are also used.
  - master branch: it is just a copy of the best performing branch the bounding box branch.

## Documentation
In this section, I write a brief description of some of the repository files/folders

#### lstm.py
In this file, I build the LSTM network and do the training, validation and testing on the dataset.

#### preprocessing.py
In this file, I preprocess the data, the output of MRCNN, and put it in a format ready to be fed to the LSTM.

#### utilities.py
In this file, I write some helpful functions that are used in lstm.py

#### models
In this folder, I save the best performing model weights, plots and log files of the results.

## Acknowledgement
- This project is done as part of the final project for Vision and Perception course taught by prof. Fiora Pirri - Sapienza Università di Roma.
- This project is done above the work done by my teammates [Mohamad Farid](https://github.com/voroujak). They trained MRCNN and they did the inference on the videos and generated the dataset used to train the LSTM in this project.
- This code structure of this project is inspired by the work done by [Fredrik Gustafsson](https://github.com/fregu856) in [CS224n_project](https://github.com/fregu856/CS224n_project).


