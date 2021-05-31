
import nltk
import spacy
import language_check
import essay_grader_baseline
import pandas as pd
import numpy as np
from textblob import Word
from textblob import TextBlob
from collection import Counter
from spellchecker import Spellchecker
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

PATH = '/content/Essay_Grader/data/training_set_rel3.xls'

def main():
  df = essay_grader_baseline.load_data(PATH)
  data = df[df['essay_set']==8]
  data['essay_length'] = np.nan
  data['sentence_length'] = np.nan
  data['unique_word'] = np.nan
  data['sentence'] = np.nan
  data['average_word_length'] = np.nan
  data['sentiment'] = np.nan
  data['noun'] = np.nan
  data['verb'] = np.nan
  data['adjective'] = np.nan
  data['adverb'] = np.nan
  data['punctuatuion']
  data['spelling_error'] = np.nan
  data['grammar_error'] = np.nan
  for i in range(0,data.shape[0]):
    blob = TextBlob(str(data['essay'][i]))
    spell = SpellChecker()
    nlp = spacy.load('en')
    docx = nlp(data['essay'][i])
    sentence_len = [len(sentence.split(' ')) for sentence in blob.sentences]
    word_len = [len(word) for word in blob.words]
    nouns = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
    verb = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB']
    adj = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ']
    adv = [ token.text for token in docx if token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADV']
    punc = [ token.text for token in docx if  token.is_punct == True ]
    error = spell.unknown(blob2.words)
    tool = language_check.LanguageTool('en-US')
    g_error = tool.check(str(data['essay'][i]))
    data.at[i,'essay_length'] = len(blob.words)
    data.at[i,'sentence_length'] = sum(sentence_len)/len(sentence_len)
    data.at[i,'unique_word'] = len(set(blob.words))
    data.at[i,'sentence'] = len(blob.sentences)
    data.at[i,'average_word_length'] = sum(word_len) / len(word_len)
    data.at[i,'sentiment'] = blob.sentiment.polarity
    data.at[i,'noun'] = len(nouns)
    data.at[i,'verb'] = len(verb)
    data.at[i,'adjective'] = len(adj)
    data.at[i,'adverb'] = len(adv)
    data.at[i,'punctuatuion'] = len(punc)
    data.at[i,'spelling_error'] = len(error)
    data.at[i,'grammar_error'] = len(g_error)

  data.to_csv("essay_grader_features.csv",index=False)

if __name__ == "__main__":
  main()
