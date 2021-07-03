from os import listdir
from os.path import isfile, join
import pandas as pd
import os
import logging

from pip._internal.utils import subprocess
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
from nltk.corpus import stopwords

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Prepare-Datasets]')
logger.info(' modules imported correctly')

def download_lang(language):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", language])
download_lang('it')

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

def create_earthquake_dataset():
    mypath = 'historical_data'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    total = 0

    final = pd.DataFrame()
    counter =0
    for elem in onlyfiles:
        df = pd.read_excel(mypath + '/' + elem)
        print(' Processing file {0}, file length: {1}'.format(elem, len(df)))
        total += len(df)
        final = final.append(df)
        counter += 1
        #if counter == 2:
        #    break
        #if total > 1040000:
        #    break
    print(total)

    directory = os.path.abspath(os.getcwd())
    new_path = os.path.join(directory, "historical_data")

    try:
        new_path = os.mkdir(new_path)
        print(' Folder created')
    except FileExistsError:
        print(' Folder Already Exists')
        new_path = os.path.join(directory, "historical_data")

    new_path = os.path.join(directory, "historical_data")


    final = final.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    final = final.drop(columns=['Unnamed: 0'])
    final = final.sample(1040000)
    final = final.reset_index(drop=True)
    final.to_excel(new_path + "/historical_tweets_Full.xlsx", sheet_name='Sheet_name_1')


def create_external_dataset() :
    mypath = 'training_topics'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    total = 0

    final_External = pd.DataFrame()
    #counter = 0
    for elem in onlyfiles:
        if '_01' in elem:
            l = 0
        else:
            df = pd.read_excel(mypath + '/' + elem)
            logger.info(' Processing file {0}, file length: {1}'.format(elem, len(df)))
            total += len(df)
            final_External = final_External.append(df)
    logger.info('Total number of records in final external topic dataframe: {0}'.format(total))
    return final_External

if __name__ == '__main__' :
    logger.info(' starting process..')
    #create df with non related topics
    logger.info(' creating external topics dataset..')
    gossip = create_external_dataset()
    #print(gossip)

    #read full earthquakes df
    logger.info(' creating earthquakes dataset..')
    quakes = pd.read_excel("historical_data/historical_tweets_Full.xlsx")

    #select random sample
    quakes = quakes.sample(200000)

    #keep only content columns
    gossip = gossip['content']
    quakes = quakes['content']

    #append dataframes
    quakes = quakes.append(gossip)
    logger.info(" final length of training data: {0}".format(len(quakes)))
    papers = quakes

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
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    # Prepare data for LDA Analysis
    # start by tokenizing the text and removing stopwords.
    # Next, we convert the tokenized object into a corpus and dictionary

    logger.info(' Preparing data for LDA Analysis..')

    stop_words = stopwords.words('italian')
    stop_words.extend(['https', 'http', 'bqjkco', 'xe', 'xf', 'gi', 'pi', 'xec'])

    data = papers.content_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])

    logger.info(' Creating dictionary..')
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    # lda model training
    logger.info(' Training model..')
    # number of topics
    num_topics = 7
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Analyzing LDA model results

    # let’s visualize the topics for interpretability. To do so, we’ll use a popular visualization package, pyLDAvis which is designed to help interactively with:
    # 1 Better understanding and interpreting individual topics, and
    # 2 Better understanding the relationships between the topics.
    # For (1), you can manually select each topic to view its top most frequent and/or “relevant” terms, using different values of the λ parameter. This can help when you’re trying to assign a human interpretable name or “meaning” to each topic.
    # For (2), exploring the Intertopic Distance Plot can help you learn about how topics relate to each other, including potential higher-level structure between groups of topics

    # Visualize the topics
    logger.info(' Visualizing topics..')
    # pyLDAvis.enable_notebook() ONLY FOR NOTEBOOK
    # test
    directory = os.path.abspath(os.getcwd())
    LDAvis_data_filepath = os.path.join(directory, "historical_data")

    try:
        LDAvis_data_filepath = os.mkdir(LDAvis_data_filepath)
        logger.info(' Folder created')
    except FileExistsError:
        logger.info(' Folder Already Exists')
        LDAvis_data_filepath = os.path.join(directory, "LDAvis")

    LDAvis_data_filepath = os.path.join(directory, "LDAvis")
    # end test

    # LDAvis_data_filepath = os.path.join('ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = gensimMod.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'ldavis_prepared_' + str(num_topics) + '.html')

    # LDAvis_prepared ##only for notebook
    # test for commit 12345
    # test for commitpush





