import pandas as pd
import re
import urllib.parse




original = pd.read_csv('Georeferencing/earthquake_synonyms.csv', sep=',')

synonyms = original.term.tolist()

tweets = pd.DataFrame(columns=['tweets'],    data =['puppiniello è andato al mare',
                                                    'il piccoletto ha detto le parolacce',
                                                    'scossa vicino Rieti',
                                                    'sisma a Tripoli', "Veramente c'\xe8 stata una scossa di terremoto a Ischia? E state bene?"])

#tweets['check'] = 0


new = pd.DataFrame(columns=['tweets'])

for elem in synonyms:
       sel = tweets[tweets['tweets'].str.contains(elem, na=False, case=False)]
       new = new.append(sel)
       #for idx, row in tweets.iterrows() :
              #if row['tweets'].str.contains(elem, na=False, case=False) :
              # if row['term'] in elem:
              #        tweets.at[idx, 'check'] = 1
              #        print('Tweet matched: {0}'.format(ln['tweets']))

print(new.drop_duplicates())


df1 = pd.DataFrame(columns=['A'],    data=['z', 'c', 'g', 'h', 'k', 'm'])
df2 = pd.DataFrame(columns=['A'],    data=['c', 'h', 'm', 'o'])

df2['B'] = [10, 20, 30, 40]

x = df1.merge(df2, on='A')

print(x)

