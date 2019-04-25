#!/usr/bin/env python
# coding: utf-8

# In[93]:


import json_lines
import csv

def process_tweet(tweet):  
    d = {}
    d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['text'] = tweet['full_text']
    d['user'] = tweet['user']['screen_name']
    d['user_loc'] = tweet['user']['location']
    d['created_at'] = tweet['created_at']
    return d

if False:
    with open('congress_dataset/senators-1.jsonl', 'rb') as f:
        with open(r'senators-1-tweets.csv', 'a') as file:
            writer = csv.writer(file)
            for item in json_lines.reader(f):
                # Only collect tweets in English
                if item['lang'] == 'en' and len(item['entities']['hashtags']) > 0:
                    tweet_data = process_tweet(item)
                    writer.writerow(list(tweet_data.values()))


# In[94]:


import pandas as pd
tweets = pd.read_csv("senators-1-tweets.csv", header=None, names=['hashtags', 'text', 'user', 'user_location'])  
print('num tweets: {}'.format(len(tweets)))
tweets.head()


# In[95]:


import spacy
nlp = spacy.load('en')


# In[ ]:





# In[96]:


docs = []
N = 1000
for i in range(N):
    docs.append(nlp(tweets.iloc[i]['text']))


# In[ ]:


stop_words = ['senator']


# In[97]:


# Clean tweets here
cleaned_tweets = []
cleaned_tweets_text = []
for doc in docs:
    curr_tweet = []
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
            curr_tweet.append(w.lemma_) # add lemmatized version of the word
    cleaned_tweets.append(curr_tweet)
    cleaned_tweets_text.append(' '.join(curr_tweet))


# In[98]:


# here we should get only cleaned tweets
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

bigram = gensim.models.Phrases(cleaned_tweets)


# In[99]:


cleaned_tweets = [bigram[t] for t in cleaned_tweets]


# In[100]:


# create dictionary and corpus
dictionary = Dictionary(cleaned_tweets)
corpus = [dictionary.doc2bow(clean_tween) for clean_tween in cleaned_tweets]


# In[101]:


#### LSI MODEL basically SVD / Principal component analysis

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)


# In[102]:


lsimodel.show_topics(num_topics=5)


# In[103]:


# HDP - Hierarchical Dirichlet process 
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()


# In[123]:


# LDA 
ldamodel = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)
ldamodel.show_topics()


# In[105]:


# use lda and nmf in sklearn

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[119]:


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(cleaned_tweets_text)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(cleaned_tweets_text)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)


# In[120]:


tfidf_feature_names[2], tf_feature_names[2]


# In[121]:


for line in tf:
    print(line)
    break


# In[124]:


import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# In[ ]:




