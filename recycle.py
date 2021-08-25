# from bs4 import BeautifulSoup
# import requests
import logging
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords
import re
import nltk
import logging
nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords

# start by tokenizing the text and removing stopwords.
# Next, we convert the tokenized object into a corpus and dictionary
stop_words = stopwords.words('italian')
stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])
nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def make_bigrams(bigram_mod, texts):
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


def perform_sentiment_analysis(sentiment_dataset, tweet_dataset):
    tweets = tweet_dataset
    df = sentiment_dataset

    # give in input a list of words
    tweets['content_split'] = tweets['content_processed'].apply(lambda x: x.lower().split(' '))
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

    return tweets

    #test = df[df['category'] == 'negative']
    #print(' Done')
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(df)
    #    print(tweets[['content', 'sentiment_value', 'sentiment_unprocessed']])
    #print(tweets['sentiment_value'].min())
    #print(tweets['sentiment_value'].max())

    #print(tweets['sentiment_unprocessed'].min())
    #print(tweets['sentiment_unprocessed'].max())


def remove_url(txt):
    try:
        # txt = urllib.parse.unquote(txt)
        # txt = unidecode.unidecode(txt)
        # for key in unicodeStrings:
        #     if key in txt:
        #         txt = txt.replace(key, unicodeStrings[key])
        txt = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

    except:
        txt = txt

    return txt

