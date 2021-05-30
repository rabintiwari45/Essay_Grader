import re
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('stopwords')

import warnings
warnings.filterwarnings("ignore")

#PATH = '/content/training_set_rel3.xls'
PATH = "C:/Users/Rabin/Documents/Downloads/asap-aes/training_set_rel3.xls"
MAX_FEATURES = 1500


def load_data(path):
    filename, fileextension = os.path.splitext(path)
    if fileextension == '.csv':
        return pd.read_csv(path)
    elif fileextension == '.xlsx' or fileextension == '.xls':
        return pd.read_excel(path)
    else:
        print("Can't open the file.")
    
def bag_of_words(dataset,MAX_FEATURES):
    corpus = []
    for i in range(0,dataset.shape[0]):
        text = re.sub('[^a-zA-Z]',' ',dataset['essay'][i])
        remove_stopwords = list(stopwords.words('english'))
        remove_words = ["person", "organization", "location", "date", "time", "money", "percent", "caps","emails", "month", "num", "dr", "city", "state"]
        text =" ".join(i for i in text.lower().split() if i not in remove_words )# or if i not in remove_stopwords))
        text = " ".join(i for i in text.split() if i not in remove_stopwords)
        corpus.append(text)

    cv = CountVectorizer(max_features=MAX_FEATURES)
    return cv.fit_transform(corpus).toarray()

def normalize_label(dataset):
    dataset['domain1_score'][dataset[dataset['essay_set']==1].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==1].index]-(dataset[dataset['essay_set']==1]['domain1_score'].min()))/(dataset[dataset['essay_set']==1]['domain1_score'].max()-dataset[dataset['essay_set']==1]['domain1_score'].min())
    dataset['domain1_score'][dataset[dataset['essay_set']==2].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==2].index]-(dataset[dataset['essay_set']==2]['domain1_score'].min()))/(dataset[dataset['essay_set']==2]['domain1_score'].max()-dataset[dataset['essay_set']==2]['domain1_score'].min())
    dataset['domain1_score'][dataset[dataset['essay_set']==3].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==3].index])/(dataset[dataset['essay_set']==3]['domain1_score'].max())
    dataset['domain1_score'][dataset[dataset['essay_set']==4].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==4].index])/(dataset[dataset['essay_set']==4]['domain1_score'].max())
    dataset['domain1_score'][dataset[dataset['essay_set']==5].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==5].index])/(dataset[dataset['essay_set']==5]['domain1_score'].max())
    dataset['domain1_score'][dataset[dataset['essay_set']==6].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==6].index])/(dataset[dataset['essay_set']==6]['domain1_score'].max())
    dataset['domain1_score'][dataset[dataset['essay_set']==7].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==7].index])/(dataset[dataset['essay_set']==7]['domain1_score'].max())
    dataset['domain1_score'][dataset[dataset['essay_set']==8].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==8].index])/(dataset[dataset['essay_set']==8]['domain1_score'].max())
    label = dataset['domain1_score']*10
    label_round = label.round(0).astype(int)
    return label_round
    

def split_data(X,y,stratified_labels):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=23,stratify=stratified_labels)
    return X_train,X_test,y_train,y_test

def model():
    data = load_data(PATH)
    label_nor = normalize_label(data) 
    features = bag_of_words(data,MAX_FEATURES)
    X_train,X_test,y_train,y_test = split_data(features,label_nor,data['essay_set'])

    log_clf = LogisticRegression()
    log_clf.fit(X_train,y_train)
    y_pred = log_clf.predict(X_test)
    kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic')
    print('The kappa score is {}'.format(kappa))

if __name__ == "__main__":
  model()
    
    
    
    
        












