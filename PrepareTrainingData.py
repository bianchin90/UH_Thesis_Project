from os import listdir
from os.path import isfile, join
import pandas as pd
import os
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
