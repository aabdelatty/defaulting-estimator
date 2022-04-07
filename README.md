
# Loan Defaulting Estimator
Defaulting estimator created using Python (sklearn, pytest, fastapi, ..), ready to be shipped in Docker container.  

# Table of contents
1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Modules](#modules)
    * [Research Phase](#research-phase)
        * [Exploratory Data Analysis (EDA)](#eda)
        * [Modelling](#modelling)
    *  [Gradient Boosting Model](#gradient-boosting-model)
    *  [App](#app)
4. [Setup](#Setup)
	* [Requirements](#requirements)
	* [ Local setup](#local_setup)
	* [ Docker](#docker)

## Introduction
Financial Technologies (fintech) have been booming making financial services smoother. Fintech has a wide range of applications that includes Digital Payments, Loaning and defaulting estimation, Fraud detection and prevention, and Portfolio management. This project tackles the problem of deafaulting estimation, more specifically given user history and the info of the requested loan using Machine Learning we can predict if the user is going to default or not. 

## Technologies

- [Python](https://www.python.org/)
- [pytest](https://docs.pytest.org/en/7.1.x/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Swagger](https://swagger.io/)
- [Docker](https://www.docker.com/)
## Modules
### Research Phase
This module includes two Jupyter Notebooks:
#### EDA: 
    * Explores the data through visualization and summary statistics 
    * Tests some hypotheses about the data 
    * Explore the possibility of introducing new variables (calculated for existing ones)
    * Define the importants of varibles, and decide if they are imrortant enough to be includind in the modeling phase
    * Handeling Null values.

#### Modelling
    * feature engineering (handling Null values, introducing new variables, and deleting unimportant features) 
    * Train Gradient Boosting as the base model
    * Train both Random Forest and Gradient Boosting with tuning the class weight parameter, due to data imbalance
    * Upsample the trainig data and repeat thr previous step
    * Select the best model and parameters to be exposed as a rest api.


## Gradient Boosting Model
This package includes sub-packages and files for:
* Feature engineering 
* Pipeline training (XGBoost as classifier)
* Predictions 
* Unit tests

## App
Exposes the Gradient Boosting Model package as a rest api using FastApi
## Setup
### Requirements
All dependencies can be found in "requirements.txt"
### Local setup
To run this project locally 
```
$ cd app
$ uvicorn main:app --reload
```
### Docker
To run this project inside a docker container:
```
$ docker build -t <img_name> .
$ docker run -d --name <container_name> -p <lcl_port>:5000 <img_name>
```

