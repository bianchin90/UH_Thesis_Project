import datetime

import pandas as pd
import os
import re
# Import the wordcloud library
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models as gensimMod
import pickle
import pyLDAvis
import gensim
from gensim.utils import simple_preprocess
import nltk
import logging
nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
import datetime as dt
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [test-final-model]')
logger.info(' modules imported correctly')

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

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    return (sub_li)

if __name__ == '__main__':

    #load model

    lda = gensim.models.LdaMulticore.load('final_model/final_model.model')

    # Read data into papers
    logger.info(' Reading DF..')
    raw = pd.read_excel('historical_data/historical_tweets_2012-05-01_2012-06-01.xlsx')
    len_df = len(raw)

    #sort by date
    raw = raw.sort_values(by=['date'])

    #process in batches of 5 minutes
    time_window = 5
    raw['date'] = pd.to_datetime(raw['date'])
    #print(papers[['date', 'content']])
    last = raw.date.max()
    start = raw.date.min()
    next = start + dt.timedelta(minutes=time_window)
    print('{0}, {1}, {2}'.format(start, next, last))

    #define array x and y to plot
    x = [] #here you add the count od hearthquakes detected
    y = [] #here you add the timeshift
    #only for test
    counter = 0
    while start < last :
        print('timeframe selected: from {0} to {1}'.format(start, next))
        df = raw[(raw['date'] >= start) & (raw['date'] < next) ]
        print(len(df))

        # keep unnecessary columns
        papers = df[['content']]
        # Print out the first rows of papers
        # papers.head()

        logger.info(' Processing textual attributes..')
        # Remove punctuation/lower casing
        papers['content_processed'] = \
            papers['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
        # Convert the titles to lowercase
        papers['content_processed'] = \
            papers['content_processed'].map(lambda x: x.lower())
        # Print out the first rows of papers
        papers['content_processed'].head()

        # Exploratory Analysis
        # To verify whether the preprocessing, we’ll make a word cloud using the wordcloud package to get a visual representation of most common words.
        # It is key to understanding the data and ensuring we are on the right track, and if any more preprocessing is necessary before training the model.

        logger.info(' Generating WordCloud..')
        # Join the different processed titles together.
        long_string = ','.join(list(papers['content_processed'].values))
        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',
                              width=800, height=400)
        # Generate a word cloud
        #wordcloud.generate(long_string)
        # Visualize the word cloud
        #wordcloud.to_image()

        # Prepare data for LDA Analysis
        # start by tokenizing the text and removing stopwords.
        # Next, we convert the tokenized object into a corpus and dictionary

        logger.info(' Preparing data for LDA Analysis..')

        stop_words = stopwords.words('italian')
        stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])

        data = papers.content_processed.values.tolist()
        data_words = list(sent_to_words(data))
        # remove stop words
        data_words = remove_stopwords(data_words)
        #logger.info(data_words[:1][0][:30])

        logger.info(' Building Bi- and Tri- grams..')
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
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        #logger.info(data_lemmatized[:1])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        #feed
        #declare list containing predictions
        forecast = []
        logger.info('printing LDA predictions.. ')
        for body in corpus :
            vector = lda[body]
            #new_v = Sort(vector[0])
            new_v = sorted(vector[0], key=lambda x: x[1], reverse=True)
            if new_v[0][0] == 3:
                #print(' Hearthquake detected')
                forecast.append('Heartquake')
            else:
                forecast.append('Other')


        #conta tutti i tweet catalogati come terremoto
        #effettua  la doppia  sentiment analysis su questi
        papers['forecast'] = forecast
        n_detection = papers[papers['forecast'] == 'Heartquake']

#        print(n_detection.groupby('forecast').count())
        logger.info(' Eathquake tweets detected: {0}'.format(len(n_detection)))

        x.append(start)
        y.append(len(n_detection))

        #set next time window
        start = next
        next = next + dt.timedelta(minutes=time_window)
        counter += 1
        plt.plot(x, y)
        plt.pause(0.05)
        #if counter == 150:
        #    break
    print('done')
#plt.plot(x, y)
plt.show()
