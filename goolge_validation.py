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

def update_map(n):

    df_sub = pd.read_csv("Stream_Data/CitiesFound.csv")
    logger.info(' Updating map..')

    for idx, row in df_sub.iterrows():
        df_sub.at[idx, 'details'] =  'City: {0} <br>Tweets per 1000 inhabitants: {1} <br>Magnitude: {2}'.format(row['city'], row['tweets_norm'], row['magnitudo'])
    # Create figure
    locations = [go.Scattermapbox(
        lon=df_sub['lon'],
        lat=df_sub['lat'],
        mode='markers',
        # marker={'color': df_sub['city'], 'size': 10}, gives error
        #marker={'color': df_sub['tweets'], 'size': 10},
        marker={'color': 'green', 'size':10},
        # unselected={'marker' : {'opacity':0, 'color' : 'black'}},
        # selected={'marker': {'size': 5}},
        hoverinfo='text',
        hovertext=df_sub['details'],
        customdata=df_sub['city'],
    )]
    # Return figure
    return {
        'data': locations,
        'layout': go.Layout(
            uirevision='foo',  # preserves state of figure/map after callback activated
            #clickmode='event+select',
            hovermode='closest',
            hoverdistance=3,
            title=dict(text="Where is the earthquake?", font=dict(size=50, color='#cccccc')),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,  # orientation
                style='outdoors', # options available "basic", "streets", "outdoors", "light", "dark", "satellite", or "satellite-streets" need access token
                                           # "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner" or "stamen-watercolor" no token
                center=dict(
                    lat=42.44208797622657,
                    lon=12.966702481337714
                ),

                pitch=0,  # incidence angle
                zoom=5
            ),
            paper_bgcolor=app_settings['background_color'],

        )
    }


if __name__ == '__main__':

    #print(burst)
    burst = find_burst()
    print(burst)
