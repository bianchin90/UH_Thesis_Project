# Importing modules
import subprocess
import pandas as pd
from feel_it import EmotionClassifier, SentimentClassifier


emotion = EmotionClassifier()
sentiment = SentimentClassifier()



print(' Reading input file..')
df = pd.read_excel('historical_data/historical_tweets_2012-05-01_2012-06-01.xlsx')
df = df.dropna()
df = df['content']
df = df.sample(15)

print(' analyzing tweets..')
#emo = []
#sent = []
#for index, row in df.iterrows():
    #print(row['content'])
    #single_emo = emotion.predict(row['content'])
    #if single_emo not in emo:
        #print(' new emotion detected: {0}'.format(single_emo))
        #   emo = emo.append(single_emo)

    #    single_sent = sentiment.predict(row['content'])
    #    if single_sent not in sent:
#        print(' new sentiment detected: {0}'.format(single_sent))
#        sent = sent.append(single_sent)

#print(' total n° of sentiments detected: {0}'.format(len(emo)))
#print(emo)

#print(' total n° of emotions detected: {0}'.format(len(sent)))
#print(sent)

print(' computing emotions analysis..')
emo = emotion.predict(df.to_list())
my_emo = set(emo)
print(emo)

print(' computing sentiment analysis..')
sent = sentiment.predict(df.to_list())
my_sent = set(sent)
print(sent)


