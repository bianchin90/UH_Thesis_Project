#from bs4 import BeautifulSoup
#import requests
import logging
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords
import re


# Define functions for stopwords, bigrams, trigrams and lemmatization
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Sentiment-Analysis]')
logger.info(' modules imported correctly')

pd.set_option('display.width', 1000)


print(' Loading sentiment dataset..')
#read sentiment dataset
df = pd.read_csv('SentimentData/Sentiment_words.csv', sep=';')
#set values based on polarity
df.loc[df['category'] == 'negative', 'value'] = -1
df.loc[df['category'] == 'positive', 'value'] = 1
df.loc[df['category'] == 'neutral', 'value'] = 0

#test tweets (will be replaced by stream data)
print(' Streaming tweets..')
tweets = pd.read_excel('historical_data/historical_tweets_2012-06-01_2012-07-01.xlsx')

#test only for speedness
tweets = tweets.dropna()
tweets = tweets.sample(1000)

print(' Preprocessing data..')
# Remove punctuation/lower casing
tweets['content_processed'] = tweets['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
# Convert the titles to lowercase
tweets['content_processed'] = tweets['content_processed'].map(lambda x: x.lower())
# Print out the first rows of tweets
tweets['content_processed'].head()

# start by tokenizing the text and removing stopwords.
# Next, we convert the tokenized object into a corpus and dictionary
stop_words = stopwords.words('italian')
stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])

data = tweets.content_processed.values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)
logger.info(data_words[:1][0][:30])

# Build the bigram and trigram models
# The two important arguments to Phrases are min_count and threshold.
# The higher the values of these param, the harder it is for words to be combined
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

logger.info(' Removing stopwords..')
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#give in input a list of words
tweets['content_split'] = tweets['content_processed'].apply(lambda x : x.lower().split(' '))
data_lemmatized = lemmatization(tweets['content_split'], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

tweets['lemmatization'] = data_lemmatized
#print(tweets[['content_processed', 'lemmatization']])

tweets['sentiment_value'] = ''
tweets['sentiment_unprocessed'] = '' #test to prove that lemmatization and text processing is useful

print(' Iterating tweets..')

#iterate over tweets
for index, row in tweets.iterrows():
    tweet_content= row['content_split']
    #content_words = tweet_content.lower().split(' ')
    content_words = row['content_split']
    to_inspect = pd.DataFrame(columns=['word'], data = content_words)

    to_inspect = to_inspect.merge(df, left_on='word', right_on='word')
    #row['sentiment_value'] = to_inspect.value.sum()
    tweets.at[index, 'sentiment_value'] = to_inspect.value.sum()
    #print(to_inspect.value.sum())

    #test on unprocessed text
    tweet_content_unp= row['content']
    content_words_unp = tweet_content_unp.lower().split(' ')
    to_inspect_unp = pd.DataFrame(columns=['word_unp'], data = content_words_unp)
    to_inspect_unp = to_inspect_unp.merge(df, left_on='word_unp', right_on='word')
    tweets.at[index, 'sentiment_unprocessed'] = to_inspect_unp.value.sum()


#test = df[df['category'] == 'negative']
print(' Done')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(df)
    print(tweets[['content', 'sentiment_value', 'sentiment_unprocessed']])
print(tweets['sentiment_value'].min())
print(tweets['sentiment_value'].max())

print(tweets['sentiment_unprocessed'].min())
print(tweets['sentiment_unprocessed'].max())
