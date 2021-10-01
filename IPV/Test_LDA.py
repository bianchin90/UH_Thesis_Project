import re
# Import the wordcloud library
import time

from gensim.models import CoherenceModel
from nltk import word_tokenize
from wordcloud import WordCloud
import gensim.corpora as corpora
import gensim
import nltk
import logging

nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
import datetime as dt
import recycle
from feel_it import EmotionClassifier, SentimentClassifier
import pandas as pd
import Geoprocessing_Test as Geoprocessing

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [LDA-Test]')
logger.info(' modules imported correctly')

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

#############CODE FROM STREAMER
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'


def most_frequent(List):
    return max(set(List), key=List.count)


# if __name__ == '__main__':
def process_data(input_text):
    text = input_text
    # load model
    lda = gensim.models.LdaMulticore.load(
        'C:/Users/filip/PycharmProjects/UH_Thesis_Project/final_model/final_model.model')

    logger.info(' Processing textual attributes..')
    # Remove punctuation/lower casing
    text = re.sub('[,\\.!?]', '', str(text))
    text = text.lower()
    # papers['content_processed'] = \
    #     papers['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
    # Convert the titles to lowercase
    # papers['content_processed'] = \
    #     papers['content_processed'].map(lambda x: x.lower())
    # # Print out the first rows of papers
    # papers['content_processed'].head()

    # Prepare data for LDA Analysis
    # start by tokenizing the text and removing stopwords.
    # Next, we convert the tokenized object into a corpus and dictionary

    logger.info(' Preparing data for LDA Analysis..')

    stop_words = stopwords.words('italian')
    stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco', ','])

    data = text

    #data_words = list(recycle.sent_to_words(data))

    # remove stop words
    #data_words = recycle.remove_stopwords(data)
    text_tokens = word_tokenize(data)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    #print(tokens_without_sw)
    data_words = tokens_without_sw


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
    data_lemmatized1 = recycle.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    data_lemmatized = [x for x in data_lemmatized1 if x != []]
    #print(data_lemmatized)


    # logger.info(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # id2word = corpora.Dictionary(tokens_without_sw)
    # Create Corpus
    texts = data_lemmatized
    # texts = tokens_without_sw
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # feed
    # declare list containing predictions
    forecast = []
    logger.info(' Printing LDA predictions.. ')
    #print(len(corpus))
    eq_counter = 0
    for body in corpus:
        vector = lda[body]
        new_v = sorted(vector[0], key=lambda x: x[1], reverse=True)
        #print(new_v)
        if new_v[0][0] == 3:
            #logger.info(' the main topic is about an earthquake ')
            eq_counter += 1
        else:
            gg = 0
    if eq_counter >= 1 :
        logger.info(' the topic of this tweet might be about an earthquake ')
    else :
        logger.info(' the topic of this tweet could not be detected')


if __name__ == '__main__':
    time.sleep(3)
    test = input('Please post a tweet: ')
    while test != 'stop':
        process_data(test)
        time.sleep(2)
        test = input('Please post another tweet: ')
