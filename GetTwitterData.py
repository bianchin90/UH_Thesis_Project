import os
import tweepy as tw
import pandas as pd

desired_width=320

pd.set_option('display.width', desired_width)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',10)

consumer_key= 'ajpBwWo9H7flJ6Kt8YlSc8swk'
consumer_secret= 'b0VXzBmVU2lSiyj3lsOlQCQHjYqOoRf3SNYhYsLiGEF0xkg5XY'
access_token= '1408154749166362632-42mW4BTiuLjlaKHnujI3XSpKiSXaK8'
access_token_secret= 'PlUQx58W0jRXPiHUISyzrJcYvhzE9dUUHuJU67Vs86sTx'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

print('authentication completed')

print('Posting a tweet from Python..')
#api.update_status("Look, this is my first automated tweet from Python!")
print( 'Tweet just posted!')

# Define the search term and the date_since date as variables
search_words = "earthquake -filter:retweets"
date_since = "2021-03-01"

# Collect tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since,
              tweet_mode='extended').items(5)

results = [status._json for status in tweets]

df = pd.DataFrame( columns = ["creation_time", "tweet_id", "full_text", "location"] )
for elem in results:
    #create array with created_at, id, full_text, location
    row = [elem['created_at'], elem['id'], elem['full_text'], elem['user']['location']]
    df_length = len(df)
    df.loc[df_length] = row
    print(elem['full_text'])

print(df)
#returns an object that you can iterate or loop over to access the data collected.
# Each item in the iterator has various attributes that you can access to get information

# Collect a list of tweets
#tweet_list = [tweet.full_text for tweet in tweets]

#get users and location
#users_locs = [[tweet.user.screen_name, tweet.user.location, tweet.text] for tweet in tweets]

#for tweet in users_locs:
#    print(tweet)

#resume from pandas dataframe