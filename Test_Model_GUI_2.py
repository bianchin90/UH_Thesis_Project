import datetime

import pandas as pd
import os
import re
# Import the wordcloud library
from gensim.models import CoherenceModel
from matplotlib.animation import FuncAnimation
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
import recycle
from feel_it import EmotionClassifier, SentimentClassifier
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
pd.set_option('display.max_columns', None)

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [test-final-model]')
logger.info(' modules imported correctly')

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    return (sub_li)

#if __name__ == '__main__':
def process_data() :
    #load model
    lda = gensim.models.LdaMulticore.load('final_model/final_model.model')

    #read sentiment dataset
    sentiment = pd.read_csv('SentimentData/Sentiment_words.csv', sep=';')

    #define emotion classifier
    emotion = EmotionClassifier()

    # Read data into papers
    logger.info(' Reading DF..')
    #raw = pd.read_excel('historical_data/historical_tweets_2012-05-01_2012-06-01.xlsx')
    raw = pd.read_excel('historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')
    len_df = len(raw)

    #sort by date
    raw = raw.sort_values(by=['date'])

    #process in batches of 5 minutes
    time_window = 60 #12 hours: 720
    raw['date'] = pd.to_datetime(raw['date'])
    #print(papers[['date', 'content']])
    last = raw.date.max()
    start = raw.date.min()
    next = start + dt.timedelta(minutes=time_window)
    print('{0}, {1}, {2}'.format(start, next, last))

    #define array x and y to plot
    x = [] #here you add the count od hearthquakes detected
    y = [] #here you add the timeshift

    #declare dataframe for emotions perceived by population
    feelings = pd.DataFrame(columns=['feelings'])

    root = tk.Tk()

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
        data_words = list(recycle.sent_to_words(data))
        # remove stop words
        data_words = recycle.remove_stopwords(data_words)
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
        data_words_nostops = recycle.remove_stopwords(data_words)
        # Form Bigrams
        data_words_bigrams = recycle.make_bigrams(bigram_mod, data_words_nostops)
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        nlp = spacy.load("it_core_news_sm", disable=['parser', 'ner'])
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = recycle.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
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
                print(' Hearthquake detected')
                forecast.append('Heartquake')
            else:
                forecast.append('Other')


        #select only tweets marked as earthquake
        papers['forecast'] = forecast
        detected = papers[papers['forecast'] == 'Heartquake']

        logger.info(' Eathquake tweets detected: {0}'.format(len(detected)))

        if len(detected) > 0:
            #run sentiment analysis
            sentiment_eval = recycle.perform_sentiment_analysis(sentiment_dataset=sentiment, tweet_dataset=detected)
            print(' Sentiment analysis results')
            print(sentiment_eval['sentiment_value'].min())
            print(sentiment_eval['sentiment_value'].max())

            print(sentiment_eval['sentiment_unprocessed'].min())
            print(sentiment_eval['sentiment_unprocessed'].max())

            print(' computing emotions analysis..')
            emotion_content = sentiment_eval['content']
            emo = emotion.predict(emotion_content.to_list())
            #my_emo = set(emo)
            sentiment_eval['emotions'] = emo
            print(emo)

            extra = {'feelings': emo}
            feelings = feelings.append(pd.DataFrame(extra))

        x.append(start)
        #y.append(len(n_detection))
        y.append(forecast.count('Heartquake'))

        #set next time window
        start = next
        next = next + dt.timedelta(minutes=time_window)

        # keep only last n records for line plot (fresherst data)
        N = 15
        # using list slicing
        # Get last N elements from list
        if len(x) > N:
            x = x[-N:]
            y = y[-N:]
            plt.clf()

        #you were following this tutorial https://datatofish.com/matplotlib-charts-tkinter-gui/
        #figure1 = plt.Figure(figsize=(5, 4), dpi=100)
        # guarda questo tutorial https://towardsdatascience.com/plotting-live-data-with-matplotlib-d871fac7500b
#        figure1 = plt.figure(1)
#        ax = plt.subplot(121)
#        ax1 = plt.subplot(122)
#        ax.set_facecolor('#DEDEDE')
#        ax1.set_facecolor('#DEDEDE')

        # clear axis
        ax.cla()
        ax1.cla()
        ax.plot(x,y)
        ax1.plot(y,x)


        counter += 1
        if counter == 10:
            break
    print('done')


if __name__ == '__main__':
    figure1 = plt.figure(1, figsize=(8, 8))
    ax = plt.subplot(121)
    ax1 = plt.subplot(122)
    ax.set_facecolor('#DEDEDE')
    ax1.set_facecolor('#DEDEDE')

    # animate
    ani = FuncAnimation(figure1, process_data(), interval=1000)

    plt.show()

