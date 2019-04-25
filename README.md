# Twitter Hashtag Prediction

## Idea

The idea is actually quite straight forward. I want to predict hashtags from twitter for any text. Especially interesting would be to predict speeches from politicians and tag them. Then compare the tags on the speeches with current tags on twitter and figure out how they influence each other. 

## Method

So because I have massive amounts of data and I would really want to try out variational inference i will use this method to infere my statistical model. For starters i will use an LDA model  then distribute the hash tags into the the found topics to get some kind of tagging mechanism. Then i can get the topic distribution on new text and with that the hash tags.  Despite LDA i will use a "naive bayesian" model that only allows 1 topic for each document which maybe is better for short twitter text. 

Since one reads a lot about the probabilistic programming languages lately i will use pyro which is developed by uber and builds upon pytorch. 

## Files

400k tweets annotated with hastags

[senators-1-tweets.csv](https://github.com/SimiPro/robotjudge/blob/master/senators-1-tweets.csv) 

lda with pymc to compare against

[lda_pymc.ipynb](https://github.com/SimiPro/robotjudge/blob/master/lda_pymc.ipynb)

lda with pyro which will probably going to be my main file for starters it currently contains only naive bayesian inference

[lda_pyro.ipynb](https://github.com/SimiPro/robotjudge/blob/master/lda_pyro.ipynb)

