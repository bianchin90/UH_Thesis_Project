#tutorial 2, evauluate models, from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# Importing modules
import pandas as pd
import os
import re
# Import the wordcloud library
from wordcloud import WordCloud
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models as gensimMod
import pickle
import pyLDAvis
import logging
import gensim
from gensim.utils import simple_preprocess
import nltk
import spacy
import subprocess
import sys
from pprint import pprint
from gensim.models import CoherenceModel
import numpy as np
import tqdm

#nltk.download('stopwords')
from nltk.corpus import stopwords

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Evaluate-LDA]')
logger.info(' modules imported correctly')

logging.getLogger('nltk_data').setLevel(logging.ERROR)
logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('gensim.similarities').setLevel(logging.ERROR)

def download_lang(language):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", language])
#download_lang('it')

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()

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


if __name__ == '__main__':
    # Read data into papers
    logger.info('reading dataset..')
    papers = pd.read_excel('historical_data_IPR/historical_tweets_ML.xlsx')

    #SAMPLE ONLY FOR TESTING##################################################
    papers = papers.sample(4000)

    # keep unnecessary columns
    papers = papers[['content']]
    # Print out the first rows of papers
    #papers.head()


    # Remove punctuation/lower casing
    logger.info(' processing text..')
    papers['content_processed'] = \
    papers['content'].map(lambda x: re.sub('[,\\.!?]', '', str(x)))
    # Convert the titles to lowercase
    papers['content_processed'] = \
    papers['content_processed'].map(lambda x: x.lower())
    # Print out the first rows of papers
    papers['content_processed'].head()

    #Exploratory Analysis
    #To verify whether the preprocessing, we’ll make a word cloud using the wordcloud package to get a visual representation of most common words.
    #It is key to understanding the data and ensuring we are on the right track, and if any more preprocessing is necessary before training the model.


    # Join the different processed titles together.
    long_string = ','.join(list(papers['content_processed'].values))
    # # Create a WordCloud object
    # wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800, height=400)
    # # Generate a word cloud
    # wordcloud.generate(long_string)
    # # Visualize the word cloud
    # wordcloud.to_image()

    #Prepare data for LDA Analysis
    #start by tokenizing the text and removing stopwords.
    #Next, we convert the tokenized object into a corpus and dictionary

    stop_words = stopwords.words('italian')
    stop_words.extend(['https','http', 'bqjkco', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])

    data = papers.content_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    logger.info(data_words[:1][0][:30])

    logger.info(' Building Bi- and Tri- grams..')
    # Build the bigram and trigram models
    #The two important arguments to Phrases are min_count and threshold.
    #The higher the values of these param, the harder it is for words to be combined
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
    #print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    # print(corpus[:1])

    # build Model
    # We have everything required to train the base LDA model. In addition to the corpus and dictionary, you need to provide the number of topics as well.
    # Apart from that, alpha and eta are hyperparameters that affect sparsity of the topics. According to the Gensim docs, both defaults to 1.0/num_topics prior
    # (we’ll use default for the base model).
    # chunksize controls how many documents are processed at a time in the training algorithm.
    # Increasing chunksize will speed up training, at least as long as the chunk of documents easily fit into memory.
    # passes controls how often we train the model on the entire corpus (set to 10). Another word for passes might be “epochs”.
    # iterations is somewhat technical, but essentially it controls how often we repeat a particular loop over each document.
    # It is important to set the number of “passes” and “iterations” high enough.

    # Build LDA model
    # logger.info(' Building model..')
    # lda_model = gensim.models.LdaMulticore(corpus=corpus,
    #                                       id2word=id2word,
    #                                       num_topics=10,
    #                                       random_state=100,
    #                                       chunksize=100,
    #                                       passes=10,  # remember to set it higher
    #                                       per_word_topics=True)0

    # logger.info(' Printing keyword topics..')
    # # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # #doc_lda = lda_model[corpus]
    #
    # logger.info(' Computing coherence score..')
    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # logger.info('\nCoherence Score: ', coherence_lda)

    #hyperparameter tuning
    #hyperparameters are the settings for a machine learning algorithm that are tuned by the data scientist before training.
    # Examples would be the number of trees in the random forest, or in our case, number of topics K
    # Model parameters can be thought of as what the model learns during training, such as the weights for each word in a given topic
    #perform a series of sensitivity tests to help determine the following model hyperparameters:
    #1 Number of Topics (K)
    #2 Dirichlet hyperparameter alpha: Document-Topic Density
    #3 Dirichlet hyperparameter beta: Word-Topic Density
    # perform these tests in sequence, one parameter at a time by keeping others constant and run them over
    # the two different validation corpus sets. I’ll use C_v as choice of metric for performance comparison
    # C_v measure is based on a sliding window, one-set segmentation of the top words and
    # an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity

    logger.info(' Starting hyperparameter tuning..')
    grid = {}
    grid['Validation_Set'] = {}
    # Topics range
    min_topics = 2
    max_topics = 10
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
        # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
        gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
        corpus]
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=540)

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        print('{0}, {1}, {2}, {3}'.format(corpus_sets[i], k, a, b))
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                      k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)
                        input('press any key to continue')
        pd.DataFrame(model_results).to_csv('lda_tuning_results2.csv', index=False)
        pbar.close()
