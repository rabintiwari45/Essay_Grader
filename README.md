# Essay Grader

### THE ARTICLE FOR THE PROJECT CAN BE FOUND [HERE](https://rabintiwari45.github.io/Portfolio/post/project-3/)

## Table of Content
* Demo
* Data Source
* Overview
* Installation
* Running the program


## Demo
![image](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo.png)
![image1](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo1.png)
![image2](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo2.png)
![image3](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo3.png)
![image4](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo4.png)
![image7](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/demo7.png)

## Data Source
The data for this project is obtained from a past kaggle competition. The dataset consists approximately 13000 essays across 8 prompts.
The table below shows the description of the dataset.

![description](https://github.com/rabintiwari45/Essay_Grader/blob/main/images/essay_summary.png)

The original dataset can be found [Here](https://www.kaggle.com/c/asap-aes)

## Overview
This is a essay scoring web application trained on top of keras API. We implemented a simple bag of words model as our baseline and improve upon that.
We trained different machine learning algorithm using the features extracted from training essay. We also explored the neural approach to essay grader(Using the word embedding to train various architecture of Neural Network).

## Installation
The code is written in Python 3.7.10. The required packages and libraries for this project are:
```
nltk==3.3
spacy==3.0.6
language_check==1.1
pandas>=0.23.0
numpy>=1.19.3
textblob==0.15.3
pyspellchecker==0.5.0
gensim==3.4.0
Flask==0.10.1
gunicorn==20.1.0
itsdangerous==0.24
Jinja2==2.10
tensorflow-cpu>=2.3.1
Werkzeug==0.14.1
MarkupSafe==0.23
```
You can install all the library by running below command after cloning the repository.
```
pip install -r requirements.txt
```

## Running the program

After cloning the repository, you can run below command to run the program.

TO RUN BASELINE MODEL
```
python essay_grader_baseline.py
```
TO RUN FEATURE EXTRACTION
```
python essay_grader_feature_extraction.py
```
TO RUN MACHINE LEARNING MODEL
```
python essay_grader_machine_learning.py
```
TO RUN NEURAL NETWORK
```
python essay_grader_neural.py
```
TO TEST THE ESSAY
```
python essay_grader_test.py
```
TO RUN THE FLASK WEB APP
```
python app.py
```

### TO KNOW MORE ABOUT THE PROJECT YOU CAN CHECK THE ARTICLE [HERE](https://rabintiwari45.github.io/Portfolio/post/project-3/)




