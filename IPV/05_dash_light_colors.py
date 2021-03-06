# ------------------------------------------------------delete this section and add your own mapbox token below
# import yaml #(pip install pyyaml)
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
# mapbox_access_token = cfg['mysql']['key']
# -------------------------------------------------------
import base64
import logging
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

# set logger
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(' UH MSc [Dashboard]')
logger.info(' modules imported correctly')
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('numexpr').setLevel(logging.ERROR)

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]

# df = pd.read_excel("Georeferencing/sample_output.xlsx")
df = pd.read_csv("Stream_Data/CitiesFound.csv")

# emotions dataframe
emotions = pd.read_csv("Stream_Data/SentimentResults.csv", sep=',')


# function to get the dataframe updated
def getData():
    data = pd.read_csv("Stream_Data/CitiesFound.csv")
    df_table = data.sort_values(by='tweets_norm', ascending=False)
    df_table = df_table[['city', 'tweets_norm']]
    df_table['tweets_norm'] = df_table['tweets_norm'].round(decimals=3)

    reg_list = ["Valle d'Aosta", "Piemonte", "Liguria", "Lombardia", "Trentino-Alto Adige",
                "Veneto", "Friuli-Venezia Giulia", "Emilia Romagna", "Toscana", "Umbria",
                "Marche", "Lazio", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata",
                "Calabria", "Sicilia", "Sardegna"]
    # remove regions
    for region in reg_list:
        indexNames = df_table[df_table['city'] == region].index
        # Delete these row indexes from dataFrame
        df_table.drop(indexNames, inplace=True)
    # df_table = df_table.rename(columns={'tweets_norm': 'tweets per 1000 inhabitants'})
    # if n > 0:
    logger.info(' Retrieving geoprosessing results..')
    # print(df_table)
    try:
        data = df_table.to_dict('records')
    except:
        data = df_table
    # print(data)
    return data

#function to retrieve last tweets' burst based on a threshold value
def find_burst(threshold):
    logger.info(' Looking for new bursts..')
    limit = threshold
    streamed = pd.read_csv('Stream_Data/Earthquakes_Detection.csv', sep=',')
    streamed = streamed.sort_values(by=['X'])
    temp = streamed['Y'].astype(int)
    temp = temp.pct_change()
    streamed['shift'] = temp
    streamed['shift'] = streamed['shift'].fillna(0)
    for idx, row in streamed.iterrows():
        if row['shift'] == np.inf:
            streamed.at[idx, 'shift'] = row['Y']
    streamed = streamed[streamed['shift'] >= limit]
    if len(streamed) > 0:
        sel = streamed.tail(1)
        burst = sel['X'].iloc[-1]
        burst = '{0}'.format(burst.split('+')[0])
    else:
        burst = 'No bursts detected so far'
    return burst

#function to create data for the map. takes in input the symobology color
def create_locations(marker_color):
    df_sub = pd.read_csv("Stream_Data/CitiesFound.csv")
    logger.info(' Updating map..')
    df_sub['tweets_norm'] = df_sub['tweets_norm'].round(decimals=3)

    for idx, row in df_sub.iterrows():
        df_sub.at[idx, 'details'] = 'City: {0} <br>Tweets per 1000 inhabitants: {1} <br>Magnitude: {2}'.format(
            row['city'], row['tweets_norm'], row['magnitudo'])
    # Create figure
    locations = [go.Scattermapbox(
        lon=df_sub['lon'],
        lat=df_sub['lat'],
        mode='markers',
        # marker={'color': df_sub['city'], 'size': 10}, gives error
        # marker={'color': df_sub['tweets'], 'size': 10},
        marker={'color': marker_color, 'size': 10},
        # unselected={'marker' : {'opacity':0, 'color' : 'black'}},
        # selected={'marker': {'size': 5}},
        hoverinfo='text',
        hovertext=df_sub['details'],
        customdata=df_sub['city'],
    )]
    return locations

#funtion to create the map, takes in input data and map style
def return_map(localities, style):
    new_map = {
        'data': localities,
        'layout': go.Layout(
            uirevision='foo',  # preserves state of figure/map after callback activated
            # clickmode='event+select',
            hovermode='closest',
            hoverdistance=3,
            title=dict(text="Where is the earthquake?", font=dict(size=50, color='#cccccc')),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,  # orientation
                style=style,
                # options available "basic", "streets", "outdoors", "light", "dark", "satellite", or "satellite-streets" need access token
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
    return new_map


# define external stylesheets for css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
# bootsrap themes https://bootswatch.com/default/
# external_stylesheets=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

blackbold = {'color': 'black', 'font-weight': 'bold'}


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
            ]
        ),
    ]
    return card_content


def setCardIcon(className, color):
    icon = dbc.Card(
        html.Div(className=className, style=card_icon),
        className="bg-primary",
        style={"maxWidth": 75},
        color=color
    )
    return icon


def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:images/png;base64,' + base64.b64encode(image).decode('utf-8')


app_settings = {'background_color': '#404040', 'div_text': '#cccccc'}

app.layout = html.Div([
    # ---------------------------------------------------------------
    ####################??html.H1("Quickker - Faster is Better"),
    # Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
    # Map
    dcc.Interval(
        id='interval-component',
        interval=1 * 10000,  # in milliseconds
        n_intervals=0
    ),

    html.Div(children=[
        html.Br(),
        dbc.Row([
            html.Img(src=b64_image("C:/Users/filip/PycharmProjects/UH_Thesis_Project/images/logo2.png"),
                     style={"margin-right": "10vh"}),
            html.H4("Argo", className="card-title",
                    style={'font-size': 64, 'text-align': 'center', 'color': "black"}),
            # color app_settings['div_text']
            html.Img(src=b64_image("C:/Users/filip/PycharmProjects/UH_Thesis_Project/images/logo2.png"),
                     style={"margin-left": "10vh"}),
        ], className="mb-4", style={"margin-left": "35%"}),
        # html.Br()
    ], style={'backgroundColor': '#00b300'}),
    # html.H4("App Name", className="card-title",
    #         style={'font-size': 64, 'text-align': 'center', 'color': app_settings['div_text']}),

    html.Br(),
    html.Br(),
    # test with multiple cards
    html.Div(children=[
        dbc.Row([  # card,
            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Total N?? of tweets detected', '0'), inverse=True, id='card-counter'),
                 setCardIcon('fa fa-notes-medical', 'primary')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Severity code', 'Unknown'), inverse=True, id='severity-counter'),
                 setCardIcon('fa fa-exclamation-triangle', 'danger')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('City mostly involved', 'No results so far'), inverse=True, id='city-counter'),
                 setCardIcon('fa fa-map-marker-alt', 'success')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('More widespread sentiment', 'No idea'), inverse=True, id='sentiment-counter'),
                 setCardIcon('fa fa-meh', 'warning')], className="mt-4 shadow", ))

        ],
            className="mb-4", style={"margin-left": "2vh", "margin-right": "2vh"}),
    ]),

    dbc.Row([  # card,
        dbc.Col(),
        dbc.Col(dbc.CardGroup(
            [dbc.Card(return_card('last burst detected', 'unknown'), inverse=True, id='card-burst'),
             setCardIcon('fa fa-chart-line', 'primary')], className="mt-4 shadow", )),
        dbc.Col()
    ],
        className="mb-4", style={"margin-left": "2vh", "margin-right": "2vh"}, justify="center", ),

    html.Br(),

    dbc.Row(children=[
        dbc.Col(
            dcc.Graph(id='map',
                      # style={'display': 'inline-block'},
                      style={'display': 'inline-block', 'height': '100vh', 'width': '150vh'},
                      config={'displayModeBar': False, 'scrollZoom': True},
                      # style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                      #        # , 'width':'180vh'
                      #        },
                      # animate=True disable it to pop up automatically markers
                      )),
        dbc.Col(
            [html.Div('Cities impacted', style={'color': app_settings['div_text'], 'fontSize': 50}),
             html.Br(),
             dash_table.DataTable(
                 id='count-table',
                 # columns=[{"name": i, "id": i} for i in df.columns],
                 columns=[{'name': 'City', 'id': 'city'},
                          {'name': 'Tweets per \n1000 inhabitants', 'id': 'tweets_norm'}],
                 # title='Cities impacted',
                 data=getData(),
                 style_cell={'textAlign': 'left', 'font-family': 'sans-serif', 'border': '2px solid black',
                             'whiteSpace': 'pre-line'},
                 # style_cell_conditional=[
                 #        {
                 #            'if': {'column_id': 'city'},
                 #            'textAlign': 'left'
                 #        } for c in (['City', 'Tweets'])
                 #    ],
                 style_header={
                     'backgroundColor': '#00b300',  # 262626
                     'fontWeight': 'bold',
                     'color': 'white'
                 },
                 # style_data = {'backgroundColor': '#99e600'},
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

    ], style={'background-color': 'dark-gray', "margin-left": "2vh", "margin-right": "2vh"}
    ),

    # test button
    dbc.Row(children=[
        # dbc.Col(),
        dbc.Col([
            html.Div('Set background map',
                     style={'color': app_settings['div_text'], 'fontSize': 30, 'margin-left': '14vh'}),
            # 'text-align':'center'
            html.Br(),
            dcc.RadioItems(id='background-button',
                           options=[{'label': 'Outdoors', 'value': 'outdoors'},
                                    {'label': 'Dark', 'value': 'dark'},
                                    {'label': 'Satellite', 'value': 'satellite'},
                                    {'label': 'Streets', 'value': 'streets'},
                                    {'label': 'Light', 'value': 'light'},
                                    {'label': 'Basic', 'value': 'basic'}
                                    ],
                           value='outdoors',
                           inputStyle={'color': app_settings['div_text'], 'margin-left': '14vh',
                                       'text': 'Set background map'},
                           labelStyle={'display': 'inline-block', 'color': app_settings['div_text'],
                                       'top-margin': '25px', "width": "50%"}
                           )

        ]),
        dbc.Col([
            html.Div('Set map marker color',
                     style={'color': app_settings['div_text'], 'fontSize': 30, 'margin-left': '5vh'}),
            # 'text-align':'center'
            html.Br(),
            dcc.RadioItems(id='marker-button',
                           options=[{'label': 'Yellow', 'value': 'yellow'},
                                    {'label': 'Red', 'value': 'red'},
                                    {'label': 'Blue', 'value': 'blue'},
                                    {'label': 'Orange', 'value': 'orange'},
                                    {'label': 'Green', 'value': 'green'},
                                    {'label': 'White', 'value': 'white'}
                                    ],
                           value='green',
                           inputStyle={'color': app_settings['div_text'], 'margin-left': '5vh',
                                       'text': 'Set background map'},
                           labelStyle={'display': 'inline-block', 'color': app_settings['div_text'],
                                       'top-margin': '25px', "width": "50%"}
                           )
        ])
    ]),
    html.Br(),
    html.Br(),
    # end test

    dbc.Row(children=[
        dbc.Col(
            [html.Div('Number of tweets detected',
                      style={'color': app_settings['div_text'], 'fontSize': 50, 'text-align': 'center'}),
             dcc.Graph(id='line-chart',
                       style={'display': 'inline-block', 'height': '60vh', 'width': '100vh'},
                       animate=True
                       )],  # TEST WITH ANOTHER CHART
        ),
        dbc.Col(
            [html.Div("Users' feelings",
                      style={'color': app_settings['div_text'], 'fontSize': 50, 'text-align': 'center'}),
             dcc.Graph(id='pie-chart')])
    ], style={"margin-left": "2vh", "margin-right": "2vh"}),

    html.Br(),
    html.Br()
],
    className='ten columns offset-by-one',  # you have a total of 12 columns
    style={'backgroundColor': app_settings['background_color']}
)

#Start callbacks
# -----------------------------------------------------------------
# line chart
dff = pd.DataFrame(columns=['X', 'Y'])


@app.callback(Output('line-chart', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_line(n2):
    logger.info(' Updating line graph..')
    time = len(dff)
    try:
        streamed = pd.read_csv('Stream_Data/Earthquakes_Detection.csv', sep=',')
    except:
        streamed = pd.DataFrame(columns=['X', 'Y'])
    if len(streamed) > 0:
        new_streamed = streamed[time:]

        for index, row in new_streamed.iterrows():
            if index >= time:
                dff.loc[index] = [row['X'], row['Y']]

    trace = plotly.graph_objs.Scatter(
        x=list(dff['X']),
        y=list(dff['Y']),
        name='Scatter',
        mode='lines+markers'

    )

    return {'data': [trace],
            'layout': go.Layout(
                xaxis=dict(range=[dff['X'].min(), dff['X'].max()], title='Time', color='white'),  # gridcolor='white'
                yaxis=dict(range=[dff['Y'].min(), dff['Y'].max()], title='Count', color='white'),  # gridcolor='white'
                colorway=['#cc8500'],  # set line color
                paper_bgcolor=app_settings['background_color'],
                plot_bgcolor=app_settings['background_color']
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
    logger.info(' Updating cities table..')
    return getData()


# --------------------------------------------------------------
# update card
@app.callback(Output('card-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_card(n):
    logger.info(' Updating card of total N?? of tweets detected..')
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    maxVal = dff['Y'].sum()
    newCard = return_card("Total N?? of tweets detected", str(maxVal))
    return newCard


# --------------------------------------------------------------

# update card
@app.callback(Output('severity-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_severity(n):
    logger.info(' Updating card of severity code..')
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    severity = pd.read_csv('Stream_Data/Severity.csv', sep=',')
    if len(severity) < 50:
        newCard = return_card('Severity code', 'Unknown')
    else:
        new_val = severity['severity'].mean()
        if new_val < -0.8:
            newCard = return_card('Severity code', 'Red')
        elif (new_val >= -0.8) and (new_val < -0.50):
            newCard = return_card('Severity code', 'Orange')
        elif new_val >= -0.5:
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
    df_city = df.sort_values(by='tweets_norm', ascending=False)
    logger.info(' Updating card of most impacted city..')
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
    logger.info(' Updating card of most widespread sentiment..')
    # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    emotions = pd.read_csv("Stream_Data/SentimentResults.csv", sep=',')
    sent = emotions['feelings'].value_counts(normalize=True) * 100
    real_sent = pd.DataFrame(sent)
    real_sent = real_sent.reset_index()
    real_sent = real_sent.rename(columns={'index': 'label'})
    real_sent['feelings'] = real_sent['feelings'].round().astype(int)

    best_feel = real_sent.sort_values(by='feelings', ascending=False)
    newCard = return_card("Most widespread sentiment", best_feel['label'].iloc[0].capitalize())
    return newCard


# --------------------------------------------------------------
# update burst card
@app.callback(Output('card-burst', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_city(n):
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    burst = find_burst(threshold=0.05)
    newCard = return_card("Last burst detected", burst)
    return newCard


# --------------------------------------------------------------
# update button
@app.callback(Output('map', 'figure'), [Input('background-button', 'value'),
                                        Input('marker-button', 'value'),
                                        Input('interval-component', 'n_intervals')])
def change_button_style(value, color, n):
    markers_list = ["red", "yellow", "blue", "green", "white", "orange"]
    for elem in markers_list:
        if color == elem:
            color_choice = elem
    locations = create_locations(color_choice)

    backgrounds = ["dark", "satellite", "basic", "streets", "outdoors", "light"]
    for elem in backgrounds:
        if value == elem:
            choice = elem
    map = return_map(locations, choice)
    return map


# --------------------------------------------------------------
# update pie chart
@app.callback(Output('pie-chart', 'figure'),
              Input('interval-component', 'n_intervals')
              )
def update_pie(n):
    logger.info(' Updating pie chart..')
    emotions = pd.read_csv("Stream_Data/SentimentResults.csv", sep=',')
    sent = emotions['feelings'].value_counts(normalize=True) * 100

    real_sent = pd.DataFrame(sent)
    real_sent = real_sent.reset_index()
    real_sent = real_sent.rename(columns={'index': 'label'})
    real_sent['feelings'] = real_sent['feelings'].round().astype(int)

    pie_chart = px.pie(
        data_frame=real_sent,
        values='feelings',
        names='label',
        color='label',  # differentiate markers (discrete) by color
        color_discrete_sequence=["red", "green", "blue", "orange"],  # set marker colors
        # color_discrete_map={"WA":"yellow","CA":"red","NY":"black","FL":"brown"},
        hover_name='feelings',  # values appear in bold in the hover tooltip
        # hover_data=['positive'],            #values appear as extra data in the hover tooltip
        # custom_data=['total'],              #values are extra data to be used in Dash callbacks
        labels={"label": "Feeling"},  # map the labels
        # title="Users' feelings",  # figure title
        # template='plotly_dark',  # 'ggplot2', 'seaborn', 'simple_white', 'plotly',
        # 'plotly_white', 'plotly_dark', 'presentation',
        # 'xgridoff', 'ygridoff', 'gridon', 'none'
        width=800,  # figure width in pixels
        height=600,  # figure height in pixels
        hole=0.5,  # represents the hole in middle of pie
    )

    pie_chart.update_layout({
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "font": {"color": "white"},  # legend

    })
    logger.info('-----------------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------------')

    return pie_chart


if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=True)

# modal popup
# https://community.plotly.com/t/any-way-to-create-an-instructions-popout/18828/2
