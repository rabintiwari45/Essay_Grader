import re
import logging
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
import nltk
import essay_grader_machine_learning
import essay_grader_baseline
from nltk.tokenize import word_tokenize
from gensim.models import word2vec
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout,Flatten,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

PATH = "/content/Essay_Grader/data/essay_grader_features_extracted.csv"
NUM_FEATURES = 300
CONTEXTS = 10
MIN_WORD_COUNT = 10
DOWNSAMPLING = 1e-3
NUM_WORKER = 4
SG = 0
INPUT_DIM = 313
DROPOUT = 0.5
LEARNING_RATE_FEED = 0.001
LEARNING_RATE_LSTM = 0.0002
LEARNING_RATE_GRU = 0.0008
PATIENCE = 40
EPOCHS = 200
BATCH_SIZE = 256


def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ",str(essay_v))
    words = essay_v.lower().split()    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def essay_to_sentences(essay_v, remove_stopwords):
    raw_sentences = tokenizer.tokenize(essay_v)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def get_sentence(essay):
  sentences = []
  for essay_v in essay:
    sentences += essay_to_sentences(str(essay_v), remove_stopwords = True)
  return sentences

def word2vec1(essay,num_features,min_word_count,num_workers,context,downsampling,sg):
  sentences = get_sentence(essay)
  model_sg = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling,sg=1)
  model_cbow = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
  if sg == 1:
    return model_sg,None
  else:
    return model_sg,model_cbow

    

def makeFeatureVec(words,num_features, model_skip=None,model_cbow=None):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    index2word_set = set(model_skip.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            if model_skip is None:
              vec_cbow = model_cbow[word]
              featureVec = np.add(featureVec,vec_cbow)
            elif model_cbow is None:
              vec_skip = model_skip[word]
              featureVec = np.add(featureVec,vec_skip)
            else:
              vec_skip = model_skip[word]
              vec_cbow = model_cbow[word]
              vec = vec_skip + vec_cbow
              featureVec = np.add(featureVec,vec)

         
    featureVec = featureVec/nwords
    return featureVec

def getAvgFeatureVecs(essays, num_features,model_skip=None,model_cbow=None):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[int(counter)] = makeFeatureVec(essay,num_features, model_skip,model_cbow)
        counter = counter + 1
    return essayFeatureVecs

def get_wordvectors(essay,model_skip=None,model_cbow=None):
  essays = []
  for essay_v in essay:
    essays.append( essay_to_wordlist( essay_v, remove_stopwords=True ))
  DataVecs = getAvgFeatureVecs(essays,NUM_FEATURES, model_skip,model_cbow)
  return DataVecs

def preprocess(X_train,X_test):
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  trainDataVecs = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
  testDataVecs = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
  return trainDataVecs,testDataVecs

def label_normalize(dataset):
  dataset['domain1_score'][dataset[dataset['essay_set']==1].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==1].index]-(dataset[dataset['essay_set']==1]['domain1_score'].min()))/(dataset[dataset['essay_set']==1]['domain1_score'].max()-dataset[dataset['essay_set']==1]['domain1_score'].min())
  dataset['domain1_score'][dataset[dataset['essay_set']==2].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==2].index]-(dataset[dataset['essay_set']==2]['domain1_score'].min()))/(dataset[dataset['essay_set']==2]['domain1_score'].max()-dataset[dataset['essay_set']==2]['domain1_score'].min())
  dataset['domain1_score'][dataset[dataset['essay_set']==3].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==3].index])/(dataset[dataset['essay_set']==3]['domain1_score'].max())
  dataset['domain1_score'][dataset[dataset['essay_set']==4].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==4].index])/(dataset[dataset['essay_set']==4]['domain1_score'].max())
  dataset['domain1_score'][dataset[dataset['essay_set']==5].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==5].index])/(dataset[dataset['essay_set']==5]['domain1_score'].max())
  dataset['domain1_score'][dataset[dataset['essay_set']==6].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==6].index])/(dataset[dataset['essay_set']==6]['domain1_score'].max())
  dataset['domain1_score'][dataset[dataset['essay_set']==7].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==7].index])/(dataset[dataset['essay_set']==7]['domain1_score'].max())
  dataset['domain1_score'][dataset[dataset['essay_set']==8].index] = (dataset['domain1_score'][dataset[dataset['essay_set']==8].index])/(dataset[dataset['essay_set']==8]['domain1_score'].max())
  label = dataset['domain1_score']
  return label

def get_feedforward():
  model = Sequential()
  model.add(Dense(128, input_dim=INPUT_DIM, kernel_initializer='he_normal', activation='relu'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(1))
  return model

def get_lstm():
  model = Sequential()
  model.add(LSTM(300, dropout=DROPOUT, recurrent_dropout=0, input_shape=[1,INPUT_DIM],activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(1))
  return model

def get_gru():
  model = Sequential()
  model.add(GRU(300, dropout=DROPOUT, recurrent_dropout=0, input_shape=[1,INPUT_DIM],activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,return_sequences=True))
  model.add(Dropout(DROPOUT))
  model.add(GRU(300, recurrent_dropout=0))
  model.add(Dropout(DROPOUT))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(1))
  return model


def train(algo=None):
  data = essay_grader_baseline.load_data(PATH)
  features = ['essay_length','sentence_length','unique_word','sentence','average_word_length','spelling_error','sentiment','noun','verb','adjective','adverb','punctuation','grammar_error']
  extracted_features = data[features]
  essay_set = data['essay_set']
  essay = data['essay']
  label = label_normalize(data)
  model_skip,model_cbow = word2vec1(essay,NUM_FEATURES,MIN_WORD_COUNT,NUM_WORKER,CONTEXTS,DOWNSAMPLING,SG)
  word_vector = get_wordvectors(essay,model_skip,model_cbow)
  combined_data = essay_grader_machine_learning.concat_features(word_vector,extracted_features)
  X_train,X_test,y_train,y_test = essay_grader_baseline.split_data(combined_data,label,essay_set)
  X_train,X_test = essay_grader_machine_learning.scale(X_train,X_test)
  X_train,X_test = preprocess(X_train,X_test)
  if algo == 'feedforward':
    model = get_feedforward()
    adam = Adam(lr=LEARNING_RATE_FEED)
    callback = EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)
    model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
    model.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
    y_pred = model.predict(X_test)
    y_pred = (y_pred*10).round(0).astype(int)
    y_test_kappa = (y_test*10).round(0).astype(int)
    kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
    print('The kappa score for FeedForward Network is {}'.format(kappa))

  elif algo == 'lstm':
    model = get_lstm()
    adam = Adam(lr=LEARNING_RATE_LSTM)
    callback = EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)
    model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mae'])
    model.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
    y_pred = model.predict(X_test)
    y_pred = (y_pred*10).round(0).astype(int)
    y_test_kappa = (y_test*10).round(0).astype(int)
    kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
    print('The kappa score for LSTM is {}'.format(kappa))

  elif algo == 'gru':
    model = get_gru()
    adam = Adam(lr=LEARNING_RATE_GRU)
    callback = EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)
    model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mae'])
    model.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
    y_pred = model.predict(X_test)
    y_pred = (y_pred*10).round(0).astype(int)
    y_test_kappa = (y_test*10).round(0).astype(int)
    kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
    print('The kappa score for GRU is {}'.format(kappa))

  else:
    print("The algorithm is not available.")

def train_cv(algo=None):
  data = essay_grader_baseline.load_data(PATH)
  features = ['essay_length','sentence_length','unique_word','sentence','average_word_length','spelling_error','sentiment','noun','verb','adjective','adverb','punctuation','grammar_error']
  extracted_features = data[features]
  essay_set = data['essay_set']
  essay = data['essay']
  label = label_normalize(data)
  model_skip,model_cbow = word2vec1(essay,NUM_FEATURES,MIN_WORD_COUNT,NUM_WORKER,CONTEXTS,DOWNSAMPLING,SG)
  word_vector = get_wordvectors(essay,model_skip,model_cbow)
  combined_data = essay_grader_machine_learning.concat_features(word_vector,extracted_features)
  model_feedforward = get_feedforward()
  model_lstm = get_lstm()
  model_gru = get_gru()
  skf = StratifiedKFold(n_splits=5,random_state=42)
  qwk = []
  for train,test in skf.split(combined_data,essay_set):
    X_train,X_test = combined_data.loc[train],combined_data.loc[test]
    y_train,y_test = label.loc[train],label.loc[test]
    X_train,X_test = essay_grader_machine_learning.scale(X_train,X_test)
    X_train,X_test = preprocess(X_train,X_test)
    callback = EarlyStopping(monitor='val_loss',patience=50,restore_best_weights=True)
    if algo == 'feedforward':
      adam = Adam(lr=LEARNING_RATE_FEED)
      model_feedforward.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
      model_feedforward.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
      y_pred = model_feedforward.predict(X_test)
      y_pred = (y_pred*10).round(0).astype(int)
      y_test_kappa = (y_test*10).round(0).astype(int)
      kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
      qwk.append(kappa)

    if algo == 'lstm':
      adam = Adam(lr=LEARNING_RATE_LSTM)
      model_lstm.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
      model_lstm.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
      y_pred = model_lstm.predict(X_test)
      y_pred = (y_pred*10).round(0).astype(int)
      y_test_kappa = (y_test*10).round(0).astype(int)
      kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
      qwk.append(kappa)

    if algo == 'gru':
      adam = Adam(lr=LEARNING_RATE_GRU)
      model_gru.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
      model_gru.fit(X_train, y_train,validation_data=(X_test,y_test),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[callback])
      y_pred = model_gru.predict(X_test)
      y_pred = (y_pred*10).round(0).astype(int)
      y_test_kappa = (y_test*10).round(0).astype(int)
      kappa = cohen_kappa_score(y_test_kappa,y_pred,weights='quadratic')
      qwk.append(kappa)
    

  print("The kappa for {} is {}".format(algo,np.mean(qwk)))


if __name__ == "__main__":
  train(algo='lstm')


