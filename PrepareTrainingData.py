from os import listdir
from os.path import isfile, join
import pandas as pd
import os
import logging

#set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Prepare-Datasets]')
logger.info(' modules imported correctly')

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
    gossip = create_external_dataset()
    print(gossip)

    #read full earthquakes df
    #quakes = pd.read_excel("historical_data/historical_tweets_Full.xlsx")

    #select random sample
    #quakes = quakes.sample(200000)

    #keep only content columns
    gossip = gossip['content']
    #quakes = quakes[['content']]

    #quakes = quakes.append(gossip)
    #print(len(quakes))
    print(gossip)

