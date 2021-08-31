import pandas as pd
import re
import urllib.parse




raw = pd.read_excel('historical_data/historical_tweets_Test_Amatrice.xlsx')
raw = raw.sort_values(by=['date'])
start = '2016-08-23 22:00:00+00:00'

df = raw[(raw['date'] >= start)]
x = df.head(15)
print(x.to_string())

df.to_excel('historical_data/historical_tweets_Test_Amatrice2.xlsx', index=False)