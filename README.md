# Project: From model to porduction

This is the code used in the project for the Task 1: Anomaly detection in an IoT setting (spotlight: Stream processing)

## What is it?

This project focuses in showing the continuous deployment of a machine learning model according to new data available.

## Installation

It is needed to create a conda enviroment using the command

```shell
conda create -n name_env python
```

Once created, it can be activated
```shell
conda activate name_env
```

To install the required packages
```shell
pip install -r requirements.txt
```

## Structure

This pipeline for delivering a new version of the model consists of the files data_processing.py, model_training.py, and main.py which processes the data, trains the model, and assembles this parts to work in a single command.

For activating a specific run in an experiment, the file API_connection.py can be run to perform it.

In case new data is provided, it can be saved in the data folder and update the root in the pipeline.

## Usage

It is needed to activate the env twice in separate tabs as one will be used to lunch Mlflow
```shell
mlflow ui
```

and the other to run the commnds. To run the pipeline after provided the root for the csv file.
```shell
python main.py
```

To run activate a specific run in mlflow, it is needed to pass the experiment id and the run id which can be found in the mlruns files
```shell
python API_connection.py experiment.id run.id
```

