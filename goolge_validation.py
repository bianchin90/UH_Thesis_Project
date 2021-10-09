import pandas
import pandas as pd


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

if __name__ == '__main__':
    df = pd.read_csv('Stream_Data/CitiesFound.csv')
    df = normalize_tweets(df)
    print(df.to_string())

