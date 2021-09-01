# ------------------------------------------------------delete this section and add your own mapbox token below
# import yaml #(pip install pyyaml)
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
# mapbox_access_token = cfg['mysql']['key']
# -------------------------------------------------------
import base64
import random as rd
from random import random

import pandas as pd
import numpy as np
import dash  # (version 1.0.0)
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import plotly
import plotly.offline as py  # (version 4.4.1)
import plotly.graph_objs as go
import plotly.express as px
import dash_table
import dash_bootstrap_components as dbc
from app import app

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]

#df = pd.read_excel("Georeferencing/sample_output.xlsx")
df = pd.read_csv("CitiesFound.csv")

# emotions dataframe
emotions = pd.read_csv("SentimentResults.csv", sep=',')


# function to get the dataframe updated
def getData():
    data = pd.read_csv("CitiesFound.csv")
    df_table = data.sort_values(by='tweets', ascending=False)
    df_table = df_table[['city', 'tweets']]
    # if n > 0:
    print('table')
    print(df_table)
    try:
        data = df_table.to_dict('records')
    except:
        data = df_table
    print(data)
    return data




blackbold = {'color': 'black', 'font-weight': 'bold'}


app_settings = {'background_color' : '#404040', 'div_text' :'#cccccc'}

map_layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1 * 10000,  # in milliseconds
        n_intervals=0
    ),

    dbc.Row(children=[
        dbc.Col(
            dcc.Graph(id='map',
                      # style={'display': 'inline-block'},
                      style={'display': 'inline-block', 'height': '100vh', 'width': '150vh'},
                      config={'displayModeBar': False, 'scrollZoom': True},
                      # style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                      #        # , 'width':'180vh'
                      #        },
                      #animate=True disable it to pop up automatically markers
                      ) ),
        dbc.Col(
            [html.Div('Cities impacted', style={'color': app_settings['div_text'], 'fontSize': 50}),
             html.Br(),
             dash_table.DataTable(
                 id='count-table',
                 # columns=[{"name": i, "id": i} for i in df.columns],
                 columns=[{'name': 'City', 'id': 'city'},
                          {'name': 'Tweets', 'id': 'tweets'}],
                 # title='Cities impacted',
                 data=getData(),
                 style_cell={'textAlign': 'left', 'font-family': 'sans-serif', 'border': '2px solid black'},
                 # style_cell_conditional=[
                 #        {
                 #            'if': {'column_id': 'city'},
                 #            'textAlign': 'left'
                 #        } for c in (['City', 'Tweets'])
                 #    ],
                 style_header={
                     'backgroundColor': '#00b300', #262626
                     'fontWeight': 'bold',
                     'color': 'white'
                 },
                 #style_data = {'backgroundColor': '#99e600'},
                 # style_data_conditional=[
                 #     {
                 #         'if': {'row_index': 'odd'},
                 #         'backgroundColor': '#99e600', # old one #808080
                 #
                 #     }],
                 # fixed_rows={'headers': True},
                 # style_table={'height': '80'}
                 style_table={'height': '80vh', 'overflowY': 'auto'},
                 style_as_list_view=True
             )],

        ),

    ], style={'background-color': 'dark-gray', "margin-left":"2vh", "margin-right":"2vh"}),

],
    className='ten columns offset-by-one',  # you have a total of 12 columns
    #style={'backgroundColor': app_settings['background_color']}
)


# ---------------------------------------------------------------
# Output of Graph
@app.callback(Output('map', 'figure'),
              [Input('interval-component', 'n_intervals')]
              )
def update_map(n):
    # df_sub = df[(df['tweets'].isin(chosen_boro))]
    #df_sub = df
    df_sub = pd.read_csv("CitiesFound.csv")
    #print(df_sub)

    for idx, row in df_sub.iterrows():
        df_sub.at[idx, 'details'] =  'City: {0} <br>Tweets: {1}  <br>Magnitudo: {2}'.format(row['city'], row['tweets'], row['magnitudo'])
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
            #title=dict(text="Where is the earthquake?", font=dict(size=50, color='#cccccc')),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,  # orientation
                style='dark', # options available "basic", "streets", "outdoors", "light", "dark", "satellite", or "satellite-streets" need access token
                                           # "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner" or "stamen-watercolor" no token
                center=dict(
                    lat=42.44208797622657,
                    lon=12.966702481337714
                ),

                pitch=0,  # incidence angle
                zoom=5
            ),
            # paper_bgcolor=app_settings['background_color'],

        )
    }

# -----------------------------------------------------------------
# update table
@app.callback(Output('count-table', 'data'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_table(n):
    return getData()






# modal popup
# https://community.plotly.com/t/any-way-to-create-an-instructions-popout/18828/2
