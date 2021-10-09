import pandas
import pandas as pd
import numpy as np


def prepare_data():
    df = pd.read_csv("Stream_Data/CitiesFound.csv")

    df_pop = pd.read_excel('Georeferencing/popolazione_comuni.xlsx')


    df['tweets_norm'] = df['tweets']
    # print(df)

    print(df_pop)


    for idx, row in df_pop.iterrows():
        city = row['Sesso']
        while city.startswith(" "):
            city = city[1:]
        df_pop.at[idx, 'Sesso'] = city

    df_pop = df_pop.rename({'Sesso': 'city'}, axis=1)
    df_pop = df_pop.drop(columns=['empty', 'maschi', 'femmine'])

    df_pop.to_excel('Georeferencing/popolazione_comuni_clean.xlsx')

def normalize_tweets(input_df):
    df = input_df
    pop = pd.read_excel('Georeferencing/popolazione_comuni_clean.xlsx')
    df = df.merge(pop, left_on='city', right_on='city', how='left')
    df['tweets_norm'] = (df['tweets']/df['totale']) * 1000
    df = df.sort_values(by='tweets_norm', ascending=False)
    df = df[df['tweets_norm'].notna()]
    return df

def find_burst():
    streamed = pd.read_csv('IPV/Stream_Data/Earthquakes_Detection.csv', sep=',')
    #temp = pd.DataFrame(columns=['Z'], data=streamed['Y'])
    streamed = streamed.sort_values(by=['X'])
    temp = streamed['Y'].astype(int)
    #temp +=1
    #streamed['old'] = temp
    temp = temp.pct_change()
    streamed['shift'] = temp
    streamed['shift'] = streamed['shift'].fillna(0)
    for idx, row in streamed.iterrows():
        if row['shift'] == np.inf:
            streamed.at[idx, 'shift'] = row['Y']
    streamed = streamed[streamed['shift'] >= 0.05 ]
    if len(streamed) > 0:
        sel = streamed.tail(1)
        burst = sel['X'].iloc[-1]
        burst = 'last burst detected {0}'.format(burst.split('+')[0])
    else:
        burst = 'No bursts detected so far'
    return burst

if __name__ == '__main__':

    #print(burst)
    burst = find_burst()
    print(burst)
