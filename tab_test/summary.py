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

#df = pd.read_excel("Georeferencing/sample_output.xlsx")
df = pd.read_csv("CitiesFound.csv")

dff = pd.DataFrame(columns=['X', 'Y'])

# emotions dataframe
emotions = pd.read_csv("SentimentResults.csv", sep=',')
# emotions = pd.DataFrame({'Col1': ['fear', 'joy', 'anger', 'sadness'], 'Value': [100] * 4})
# emotions['random'] = np.around(np.random.dirichlet
#                                (np.ones(emotions.shape[0]), size=1)[0],
#                                decimals=1)
# emotions['percentage'] = (emotions['Value'] * emotions['random']).astype(int)



card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

blackbold = {'color': 'black', 'font-weight': 'bold'}


#############test with multiple cards
def return_card(title, text):
    card_content = [
        # dbc.CardHeader("Card header"),
        dbc.CardBody(
            [
                html.H4(title, className="card-title", style={'color': 'black'}),
                html.P(
                    text,
                    className="card-text",
                    style={'color': 'black'}
                ),
            ], style={'height': 150}
        ),
    ]
    return card_content


def setCardIcon(className, color):
    icon = dbc.Card(
        html.Div(className=className, style=card_icon),
        className="bg-primary",
        style={"maxWidth": 75}, #"25%"
        color=color
    )
    return icon

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:images/png;base64,' + base64.b64encode(image).decode('utf-8')



app_settings = {'background_color' : '#404040', 'div_text' :'#cccccc'}

summary_layout = html.Div(children=[
        dcc.Interval(
            id='interval-component',
            interval=1 * 10000,  # in milliseconds
            n_intervals=0
        ),
        dbc.Row([  # card,
            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Total N° of tweets detected', '0'), inverse=True, id='card-counter'),
                 setCardIcon('fa fa-notes-medical', 'primary')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Severity code', 'Unknown'), inverse=True, id='severity-counter'),
                 setCardIcon('fa fa-exclamation-triangle', 'danger')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('City mostly involved', 'Cartago'), inverse=True, id='city-counter'),
                 setCardIcon('fa fa-map-marker-alt', 'success')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Most common sentiment', 'No idea'), inverse=True, id='sentiment-counter'),
                 setCardIcon('fa fa-meh', 'warning')], className="mt-4 shadow", ))

        ],
            className="mb-4", style={"margin-left":"2vh", "margin-right":"2vh"}),
    ],
    className='ten columns offset-by-one',  # you have a total of 12 columns
    #style={'backgroundColor': app_settings['background_color']}
)

    # html.Br(),


# --------------------------------------------------------------
# update card
@app.callback(Output('card-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_card(n):
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    maxVal = dff['Y'].sum()
    newCard = return_card("Total N° of tweets detected", str(maxVal))
    return newCard

# --------------------------------------------------------------

# update card
@app.callback(Output('severity-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_severity(n):
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    severity = pd.read_csv('Severity.csv', sep=',')
    if len(severity) < 50 :
        newCard = return_card('Severity code', 'Unknown')
    else:
        new_val = severity['severity'].mean()
        if new_val < -0.8 :
            newCard = return_card('Severity code', 'Red')
        elif (new_val >= -0.8) and (new_val < -0.50) :
            newCard = return_card('Severity code', 'Orange')
        elif new_val >= -0.5 :
            newCard = return_card('Severity code', 'Yellow')
    return newCard

# --------------------------------------------------------------
# update city counter
@app.callback(Output('city-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_city(n):
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    df_city = df.sort_values(by='tweets', ascending=False)
    print(df_city)
    newCard = return_card("City mostly involved", df_city['city'].iloc[0])
    return newCard

# --------------------------------------------------------------
# update sentiment counter
@app.callback(Output('sentiment-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_sentiment(n):
    # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    emotions = pd.read_csv("SentimentResults.csv", sep=',')
    sent = emotions['feelings'].value_counts(normalize=True) * 100
    real_sent = pd.DataFrame(sent)
    real_sent = real_sent.reset_index()
    real_sent = real_sent.rename(columns={'index': 'label'})
    real_sent['feelings'] = real_sent['feelings'].round().astype(int)

    best_feel = real_sent.sort_values(by='feelings', ascending=False)
    newCard = return_card("Most common sentiment", best_feel['label'].iloc[0].capitalize())
    return newCard


# modal popup
# https://community.plotly.com/t/any-way-to-create-an-instructions-popout/18828/2
