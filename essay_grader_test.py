import nltk
import spacy
import language_check
import essay_grader_neural
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from textblob import Word
from textblob import TextBlob
from collections import Counter
from spellchecker import SpellChecker
from tensorflow.keras.models import load_model
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

PATH_SKIP = 'model_skip.model'
PATH_CBOW = 'model_cbow.model'
PATH_FEEDFORWARD = 'model_feedforward.h5'
PATH_LSTM = 'model_lstm.h5'
PATH_GRU = 'model_gru.h5'
PATH_SCALER = 'scaler.pkl'
NUM_FEATURES = 300



def extract_features(essay):
    blob = TextBlob(str(essay))
    spell = SpellChecker()
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(essay)
    tool = language_check.LanguageTool('en-US')
    sentence_len = [len(sentence.split(' ')) for sentence in blob.sentences]
    word_len = [len(word) for word in blob.words]
    nouns = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
    verb = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB']
    adj = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ']
    adv = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADV']
    punc = [ token.text for token in docx if  token.is_punct == True ]
    error = spell.unknown(blob.words)
    g_error = tool.check(str(essay))
    essay_length = len(blob.words)
    sentence_length = sum(sentence_len)/len(sentence_len)
    unique_word = len(set(blob.words))
    sentence = len(blob.sentences)
    average_word_length = sum(word_len)/len(word_len)
    sentiment = blob.sentiment.polarity
    noun = len(nouns)
    verb = len(verb)
    adjective = len(adj)
    adverb = len(adv)
    punctuation = len(punc)
    spelling_error = len(error)
    grammar_error = len(g_error)
    return essay_length,sentence_length,average_word_length,spelling_error,sentiment,noun,verb,adjective,adverb,punctuation,grammar_error,unique_word,sentence



def get_features(essay,model_skip=None,model_cbow=None):
    essays = essay_grader_neural.essay_to_wordlist(essay,remove_stopwords=True)
    DataVecs = essay_grader_neural.makeFeatureVec(essays,NUM_FEATURES,model_skip,model_cbow)
    return DataVecs

def concat_features(word_vector,features):
    features = pd.concat([pd.DataFrame(word_vector),pd.DataFrame(features)],axis=0)
    return features


def test(essay,algo=None):
    features = extract_features(essay)
    features = list(features)
    model_skip = Word2Vec.load(PATH_SKIP)
    model_cbow = Word2Vec.load(PATH_CBOW)
    model_feedforward = load_model(PATH_FEEDFORWARD)
    model_lstm = load_model(PATH_LSTM)
    model_gru = load_model(PATH_GRU)
    model_scaler = pickle.load(open(PATH_SCALER,'rb'))
    word_vector = get_features(essay,model_skip,model_cbow)
    combined_data = concat_features(word_vector,features)
    combined_data = combined_data.T
    combined_data_scaler = model_scaler.transform(combined_data)
    test = np.reshape(combined_data_scaler, (combined_data_scaler.shape[0],1, combined_data_scaler.shape[1]))
    if algo == 'feedforward':
        result = model_feedforward.predict(combined_data_scaler)
        print("The score for essay is {}".format(result))
    elif algo == 'lstm':
        result = model_lstm.predict(test)
        print("The score for essay is {}".format(result))
    elif algo == 'gru':
        result = model_gru.predict(test)
        print("The score for essay is {}".format(result))
    else:
        print("The algorithm is not available")
        
if __name__ == '__main__':
    #essay = 'the essayA cow is a domestic animal. Cows are one of the most innocent animals who are very harmless. People keep cows at their homes for various benefits. Cows are four-footed and have a large body. It has two horns, two eyes plus two ears and one nose and a mouth. Cows are herbivorous animals. They have a lot of uses to mankind. In fact, farmers and people keep cows at their homes for the same purposes.'
    essay = input("enter the essay:")
    test(essay,algo='lstm')
    