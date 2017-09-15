# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:01:07 2017

@author: mahesh
"""

import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)
import matplotlib.pyplot as plt

consumer_key= 'reCwP5TnfE7eM8GOs9ODF1cHS'
consumer_secret= 'MzUn4lLTlVRdjLy6sZZZIejFUTyUkH7hNoiu55h4nMBMEvA9zc'
access_token='214751683-abEIM3ld0SMc0rrZykma2yUApAihLiVa9HWlm78n'
access_token_secret='8W2hNU9aLR1JB32FoHqhSrYArHkPolQTVeJJCD7QKMz2i'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

public_tweets = api.search('Facebook')
print public_tweets[2].text

threshold=0
pos_sent_tweet=0
neg_sent_tweet=0
for tweet in public_tweets:
    analysis=TextBlob(tweet.text)
    if analysis.sentiment.polarity>=threshold:
        pos_sent_tweet=pos_sent_tweet+1
    else:
        neg_sent_tweet=neg_sent_tweet+1
if pos_sent_tweet>neg_sent_tweet:
    print "Overall Positive"
else:
    print "Overall Negative"
    
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return
get_data('E:\\fb.csv')
plt.plot(prices)

plt.show()

def create_datasets(dates,prices):
    train_size=int(0.80*len(dates))
    TrainX,TrainY=[],[]
    TestX,TestY=[],[]
    cntr=0
    for date in dates:
        if cntr<train_size:
            TrainX.append(date)
        else:
            TestX.append(date)    
    for price in prices:
        if cntr<train_size:
            TrainY.append(price)
        else:
            TestY.append(price)
            
    return TrainX,TrainY,TestX,TestY

def predict_prices(dates,prices,x):
    TrainX,TrainY,TestX,TestY=create_datasets(dates,prices)

    TrainX=np.reshape(TrainX,(len(TrainX),1))
    TrainY=np.reshape(TrainY,(len(TrainY),1))
    TestX=np.reshape(TestX,(len(TestX),1))
    TestY=np.reshape(TestY,(len(TestY),1))
    
    #for i in range(251):
     #   print TrainX[i],TrainY[i],'\n'
    
    
    model=Sequential()
    model.add(Dense(32,input_dim=1,init='uniform',activation='relu'))
    model.add(Dense(32,input_dim=1,init='uniform',activation='relu'))
    model.add(Dense(16,init='uniform',activation='relu'))
    
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    model.fit(TrainX,TrainY,nb_epoch=100,batch_size=3,verbose=1)
    
predict_prices(dates,prices,2)

