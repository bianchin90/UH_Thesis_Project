# Importing modules
import subprocess
import pandas as pd
from feel_it import EmotionClassifier, SentimentClassifier

def download_module(package):
    subprocess.check_call([sys.executable, "pip", "install", package])

emotion = EmotionClassifier()
sentiment = SentimentClassifier()



print(' Reading input file..')
df = pd.read_excel('historical_data/historical_tweets_2012-05-01_2012-06-01.xlsx')
df = df.dropna()
df = df['content']
#df = df.sample(20)

print(' computing emotions analysis..')
emo = emotion.predict(df.to_list())
my_emo = set(emo)
print(my_emo)

print(' computing sentiment analysis..')
sent = sentiment.predict(df.to_list())
my_sent = set(sent)
print(my_sent)


