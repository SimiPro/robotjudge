#!/usr/bin/env python
# coding: utf-8

# In[46]:





# In[48]:


import spacy
import nltk

nlp = spacy.load('en_core_web_md')

def tokenize(text):
    lda_tokens = []
    tokens = nlp(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# In[49]:


import os
import json
import collections


with open('/home/simi/projects/robot/speeches_by_speaker.json', 'r') as f:
    f_json = json.load(f)
speaches_by_speaker = f_json


# In[53]:


sen_2019 = [
    "Feinstein", "King", "Murphy", "Carper", "hirono", "Warren",
"Stabenow", "klobuchar", "Wicker", "Tester", "Fischer", "Menendez",
"Heinrich", "Gillibrand", "Brown", "Casey", "Whitehouse", "Cruz",
"Sanders", "kaine", "Cantwell", "Manchin", "Baldwin", "Barrasso"
]


def write_speech_as_tw2vec_file(senator, speeches):
    date = speeches[0]
    speech = speeches[1]
    with open(f'speeches_2/tweet2vec_input_{senator}_{date}.txt', 'a') as f:
        text = speech.replace('\n', ' ')
        text = ' '.join(prepare_text(text))
    
        for i in range(0, len(text), 145):
            f.write(text[i:i+145] + '\n')

def write_speeches():
    for sen in sen_2019:
        for speaker in speaches_by_speaker.keys():
            if sen.lower() in speaker.lower():
                write_speech_as_tw2vec_file(sen, speaches_by_speaker[speaker])

write_speeches()

#list(speaches_by_speaker.keys())


# In[ ]:




