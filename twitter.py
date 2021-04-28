#neccessary imports
import tweepy 
from decouple import config 
import pandas as pd
import csv
import re 
import string
import preprocessor as p
 
#Twitter APIs  
consumer_key = config('consumer_key')
consumer_secret = config('consumer_secret')
access_key= config('access_key')
access_secret = config('access_secret')

#Auth through APIs
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
 
api = tweepy.API(auth,wait_on_rate_limit=True)
 
csvFile = open('scraped_data.csv', 'w')
csvWriter = csv.writer(csvFile)
search_words = "covid19"      # enter your words
new_search = search_words 

#Scraped data from twitter to CSV(Tweet, username) 
for tweet in tweepy.Cursor(api.search,q=new_search,count=100,
                           lang="en",
                           since_id=0).items():
    csvWriter.writerow([tweet.text.encode('utf-8'),tweet.user.screen_name.encode('utf-8')])
