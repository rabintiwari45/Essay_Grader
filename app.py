import re
import pickle
import nltk
import string
import spacy
import language_check
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob import Word
from collections import Counter
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from flask import Flask,render_template,url_for,request




nltk.download('stopwords')
#nltk.download('punkt')
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


PATH_SKIP = "E:/Python file/essay grader/app/model_skip.model"
PATH_LSTM = "E:/Python file/essay grader/app/model_lstm.h5"
PATH_SCALER = "E:/Python file/essay grader/app/scaler.pkl"
NUM_FEATURES = 300



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():

    model_skip = Word2Vec.load(PATH_SKIP)
    model_scaler = pickle.load(open(PATH_SCALER,'rb'))
    model_lstm = load_model(PATH_LSTM)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]

        def extract_features(essay):
            blob = TextBlob(str(essay))
            spell = SpellChecker()
            nlp = spacy.load('en_core_web_sm')
            docx = nlp(str(essay))
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
            corrected_text = language_check.correct(str(data),g_error)
            return essay_length,sentence_length,average_word_length,spelling_error,sentiment,noun,verb,adjective,adverb,punctuation,grammar_error,unique_word,sentence


        def essay_to_wordlist(essay_v, remove_stopwords):
            essay_v = re.sub("[^a-zA-Z]", " ",str(essay_v))
            words = essay_v.lower().split()    
            if remove_stopwords:
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]
            return words

 
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

        tool = language_check.LanguageTool('en-US')
        g_error = tool.check(str(data))
        corrected_text = language_check.correct(str(data),g_error)
        features = extract_features(data)
        features = list(features)
        essay = essay_to_wordlist(data,remove_stopwords=True)
        word_vectors = makeFeatureVec(essay,NUM_FEATURES,model_skip)
        combined_data = pd.concat([pd.DataFrame(word_vectors),pd.DataFrame(features)],axis=0)
        combined_data = combined_data.T
        combined_data_scaler = model_scaler.transform(combined_data)
        test = np.reshape(combined_data_scaler, (combined_data_scaler.shape[0],1, combined_data_scaler.shape[1]))
        my_prediction = model_lstm.predict(test)

    return render_template('result.html', prediction = my_prediction,a=features[0],b=features[-2],c=features[1],d=features[3],e=features[4],f=features[5],g=features[6],h=features[7],i=features[8],j=features[9],k=features[10],data=str(data),corrected_text = corrected_text)

if __name__ == '__main__':
	app.run(debug=True)