import pandas as pd
import numpy as np
import essay_grader_baseline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

import warnings 
warnings.filterwarnings("ignore")


PATH = "/content/essay_grader_features_full.csv"
MAX_FEATURES = 500


def concat_features(bag_of_words,extracted_features):
  combined_data = pd.concat([pd.DataFrame(bag_of_words), 
                            extracted_features], axis=1)
  return combined_data

def scale(X_train,X_test):
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  return X_train,X_test

def main(scaler=1,algo=None):
  data = essay_grader_baseline.load_data(PATH)
  features = ['essay_length','sentence_length','unique_word','sentence','average_word_length','spelling_error','sentiment','noun','verb','adjective','adverb','punctuation','grammar_error']
  extracted_features = data[features]
  essay_set = data['essay_set']
  label = essay_grader_baseline.normalize_label(data)
  bag_of_words = essay_grader_baseline.bag_of_words(data,MAX_FEATURES)
  combined_data = concat_features(bag_of_words,extracted_features)
  X_train,X_test,y_train,y_test = essay_grader_baseline.split_data(combined_data,label,essay_set)
  if scaler:
    X_train,X_test = scale(X_train,X_test)
  else:
    X_train,X_test = X_train,X_test
  if algo == 'logistic':
    log_clf = LogisticRegression()
    log_clf.fit(X_train,y_train)
    y_pred = log_clf.predict(X_test)
    kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic')
    print("The kappa score is {}".format(kappa))
    print(X_train.shape)

  elif algo == 'svm':
    log_svm = SVC()
    log_svm.fit(X_train,y_train)
    y_pred = log_svm.predict(X_test)
    kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic')
    print("The kappa score is {}".format(kappa))

  elif algo == 'knn':
    log_knn = KNeighborsClassifier(algo='kd_tree')
    log_knn.fit(X_train,y_train)
    y_pred = log_knn.predict(X_test)
    kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic')
    print("The kappa score is {}".format(kappa))

  elif algo == 'rf':
    log_rf = RandomForestClassifier()
    log_rf.fit(X_train,y_train)
    y_pred = log_rf.predict(X_test)
    kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic')
    print("The kappa score is {}".format(kappa))

  else:
    print("The algorithm is not available.")

if __name__ == "__main__":
  main(scaler=0,algo='rf')


