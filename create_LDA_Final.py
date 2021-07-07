#topic modeling with LDA- Tutorial 1 from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# Importing modules
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

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [create-final-model]')
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

if __name__ == '__main__':
    #freeze_support()

    # Read data into papers
    logger.info(' Reading DF..')
    papers = pd.read_excel('historical_data/historical_tweets_ML.xlsx')
    len_df = len(papers)
    #pepers = papers.sample(10)

    # keep unnecessary columns
    papers = papers[['content']]
    # Print out the first rows of papers
    #papers.head()

    logger.info(' Processing textual attributes..')
    # Remove punctuation/lower casing
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

    logger.info(' Generating WordCloud..')
    # Join the different processed titles together.
    long_string = ','.join(list(papers['content_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800, height=400)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    #Prepare data for LDA Analysis
    #start by tokenizing the text and removing stopwords.
    #Next, we convert the tokenized object into a corpus and dictionary

    logger.info(' Preparing data for LDA Analysis..')

    stop_words = stopwords.words('italian')
    stop_words.extend(['https', 'http', 'bqjkco', '\xe8', 'xe', 'xf', 'gi', 'pi', 'xec', 'tco'])

    data = papers.content_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    logger.info(data_words[:1][0][:30])

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
    logger.info(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    logger.info(corpus[:1])

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
    logger.info(' Building model..')
    topics = 9
    alpha = 1/topics
    beta = 1/topics
    chunk = int((len_df/100)*70)
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topics,
                                           random_state=100,
                                           chunksize=chunk,
                                           passes=100,  # remember to set it higher
                                           per_word_topics=True,
                                           alpha=alpha,
                                           eta=beta)

    logger.info(' Printing keyword topics..')
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]

    logger.info(' Computing coherence score..')
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info('\nCoherence Score: ', coherence_lda)

    #Analyzing LDA model results

    #let’s visualize the topics for interpretability. To do so, we’ll use a popular visualization package, pyLDAvis which is designed to help interactively with:
    #1 Better understanding and interpreting individual topics, and
    #2 Better understanding the relationships between the topics.
    #For (1), you can manually select each topic to view its top most frequent and/or “relevant” terms, using different values of the λ parameter. This can help when you’re trying to assign a human interpretable name or “meaning” to each topic.
    #For (2), exploring the Intertopic Distance Plot can help you learn about how topics relate to each other, including potential higher-level structure between groups of topics


    # Visualize the topics
    logger.info(' Visualizing topics..')
    #pyLDAvis.enable_notebook() ONLY FOR NOTEBOOK
#test
    directory = os.path.abspath(os.getcwd())
    LDAvis_data_filepath = os.path.join(directory, "historical_data")

    try:
        LDAvis_data_filepath = os.mkdir(LDAvis_data_filepath)
        logger.info(' Folder created')
    except FileExistsError:
        logger.info(' Folder Already Exists')
        LDAvis_data_filepath = os.path.join(directory, "LDAvis")

    LDAvis_data_filepath = os.path.join(directory, "LDAvis")
#end test


    #LDAvis_data_filepath = os.path.join('ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = gensimMod.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'ldavis_prepared_final.html')

    #save model
    lda_model.save('final_model.model')
    logger.info(' process completed')
