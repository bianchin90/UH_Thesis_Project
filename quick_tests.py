# ------------------------------------------------------delete this section and add your own mapbox token below
# import yaml #(pip install pyyaml)
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
# mapbox_access_token = cfg['mysql']['key']
# -------------------------------------------------------
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

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]

df_geo = pd.read_excel("Georeferencing/sample_output.xlsx")

# emotions dataframe
emotions = pd.DataFrame({'Col1': ['fear', 'joy', 'anger', 'sadness'], 'Value': [100] * 4})
emotions['random'] = np.around(np.random.dirichlet
                               (np.ones(emotions.shape[0]), size=1)[0],
                               decimals=1)
emotions['percentage'] = (emotions['Value'] * emotions['random']).astype(int)


# function to get the dataframe updated
def getData():
    df_table = df_geo.sort_values(by='tweets', ascending=False)
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


############
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


app.layout = html.Div([
    # ---------------------------------------------------------------
    ####################àhtml.H1("Quickker - Faster is Better"),
    # Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
    # Map
    dcc.Interval(
        id='interval-component',
        interval=1 * 10000,  # in milliseconds
        n_intervals=0
    ),

    html.H4("Quickker - Faster is Better", className="card-title",
            style={'font-size': 64, 'text-align': 'center', 'color': 'red'}),

    html.Br(),
    html.Br(),
    # test with multiple cards
    html.Div(children=[
        dbc.Row([  # card,
            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Total N° of tweets detected', '0'), inverse=True, id='card-counter'),
                 setCardIcon('fa fa-notes-medical', 'primary')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('Overall Severity', '0'), inverse=True, id='severity-counter'),
                 setCardIcon('fa fa-exclamation-triangle', 'danger')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('City mostly involved', 'Cartago'), inverse=True, id='city-counter'),
                 setCardIcon('fa fa-map-marker-alt', 'success')], className="mt-4 shadow", )),

            dbc.Col(dbc.CardGroup(
                [dbc.Card(return_card('More widespread sentiment', 'No idea'), inverse=True, id='sentiment-counter'),
                 setCardIcon('fa fa-meh', 'warning')], className="mt-4 shadow", ))

        ],
            className="mb-4"),
    ]),

    html.Br(),

    dbc.Row(children=[
        dbc.Col(
            dcc.Graph(id='graph',
                      # style={'display': 'inline-block'},
                      style={'display': 'inline-block', 'height': '100vh', 'width': '150vh'},
                      config={'displayModeBar': False, 'scrollZoom': True},
                      # style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                      #        # , 'width':'180vh'
                      #        },
                      animate=True
                      )),
        dbc.Col(
            [html.Div('Cities impacted', style={'color': 'green', 'fontSize': 50}),
             html.Br(),
             dash_table.DataTable(
                 id='count-table',
                 # columns=[{"name": i, "id": i} for i in df_geo.columns],
                 columns=[{'name': 'City', 'id': 'city'},
                          {'name': 'Tweets', 'id': 'tweets'}],
                 # title='Cities impacted',
                 data=getData(),
                 style_cell={'textAlign': 'left', 'font-family':'sans-serif', 'border': '2px solid grey'},
                 style_header={
                     'backgroundColor': 'rgb(8, 91, 94)',
                     'fontWeight': 'bold'
                 },
                 style_data_conditional=[
                     {
                         'if': {'row_index': 'odd'},
                         'backgroundColor': 'rgb(141, 239, 242)'
                     }],
             )],

        ),

    ], style={'background-color': 'dark-gray'}),

    dbc.Row(children=[
        dbc.Col(
            [html.Div('Number of tweets detected', style={'color': 'green', 'fontSize': 50, 'text-align': 'center'}),
             dcc.Graph(id='line-chart',
                       style={'display': 'inline-block', 'height': '60vh', 'width': '100vh'},
                       animate=True
                       )],  # TEST WITH ANOTHER CHART
        ),
        dbc.Col(
            [html.Div("Users' feelings", style={'color': 'green', 'fontSize': 50, 'text-align': 'center'}),
             dcc.Graph(id='pie-chart')])
    ]),

    html.Br(),
    html.Br()
],
    className='ten columns offset-by-one',  # you have a total of 12 columns
    #style={'backgroundColor': 'black'}
)


# ---------------------------------------------------------------
# Output of Graph
@app.callback(Output('graph', 'figure'),
              [Input('interval-component', 'n_intervals')]
              )
def update_map(n):
    # df_sub = df[(df['tweets'].isin(chosen_boro))]
    df_sub = df

    print(df_sub)
    if 'Bari' not in df_sub.values:
        print('adding new city')
        df_sub.loc[len(df_sub)] = ['Bari', '41.11336132309502', '16.860921999063116', 2]
    else:
        print('Detected new tweet in existing city')
        values = [0, 1, 2, 3]
        cities = df_sub.city.unique()
        mask = (df_sub['city'] == rd.choice(cities))
        df_sub['tweets'][mask] += rd.choice(values)
    print(n)

    # Create figure
    locations = [go.Scattermapbox(
        lon=df_sub['lon'],
        lat=df_sub['lat'],
        mode='markers',
        marker={'color': df_sub['tweets'], 'size': 10},
        # marker={'color': 'yellow', 'size':10},
        # unselected={'marker' : {'opacity':0, 'color' : 'black'}},
        # selected={'marker': {'size': 5}},
        hoverinfo='text',
        hovertext=df_sub['tweets'],
        customdata=df_sub['city'],

    )]

    # Return figure
    return {
        'data': locations,
        'layout': go.Layout(
            uirevision='foo',  # preserves state of figure/map after callback activated
            clickmode='event+select',
            hovermode='closest',
            hoverdistance=2,
            title=dict(text="Where is the earthquake?", font=dict(size=50, color='green')),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,  # orientation
                style='dark',
                center=dict(
                    lat=42.44208797622657,
                    lon=12.966702481337714
                ),

                pitch=0,  # incidence angle
                zoom=5
            ),
        )
    }


# -----------------------------------------------------------------
# test with line chart
dff = pd.DataFrame(columns=['X', 'Y'])


@app.callback(Output('line-chart', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_line(n2):
    time = len(dff)
    dff.loc[len(dff)] = [time, rd.randint(1, 101)]
    print(dff)
    trace = plotly.graph_objs.Scatter(
        x=list(dff['X']),
        y=list(dff['Y']),
        name='Scatter',
        mode='lines+markers'
        # title='Number of tweets detected'
    )
    # fig = px.line(dff, x=dff['X'], y=dff['Y'])
    # fig = fig.update_layout(yaxis={'title':'Count'}, xaxis={'title':'Timeline'},
    #                   title={'text':'Number of tweets detected',
    #                   'font':{'size':28},'x':0.5,'xanchor':'center'})

    return {'data': [trace],
            'layout': go.Layout(
                xaxis=dict(range=[dff['X'].min(), dff['X'].max()]),
                yaxis=dict(range=[dff['Y'].min(), dff['Y'].max()]))

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
# update city counter
@app.callback(Output('city-counter', 'children'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
              )
def update_city(n):
    #     # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    df_city = df_geo.sort_values(by='tweets', ascending=False)
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
    best_feel = emotions.sort_values(by='percentage', ascending=False)
    newCard = return_card("Most widespread sentiment", best_feel['Col1'].iloc[0])
    return newCard


# --------------------------------------------------------------

# --------------------------------------------------------------
# update pie chart
@app.callback(Output('pie-chart', 'figure'),
              Input('interval-component', 'n_intervals')
              )
def update_pie(n):
    # link for update https://stackoverflow.com/questions/66550872/dash-plotly-update-cards-based-on-date-value
    # emotions = pd.DataFrame({'Col1': ['fear', 'joy', 'anger', 'sadness'], 'Value': [100] * 4})
    emotions['random'] = np.around(np.random.dirichlet
                                   (np.ones(emotions.shape[0]), size=1)[0],
                                   decimals=1)
    emotions['percentage'] = (emotions['Value'] * emotions['random']).astype(int)

    pie_chart = px.pie(
        data_frame=emotions,
        values='percentage',
        names='Col1',
        color='Col1',  # differentiate markers (discrete) by color
        color_discrete_sequence=["red", "green", "blue", "orange"],  # set marker colors
        # color_discrete_map={"WA":"yellow","CA":"red","NY":"black","FL":"brown"},
        hover_name='random',  # values appear in bold in the hover tooltip
        # hover_data=['positive'],            #values appear as extra data in the hover tooltip
        # custom_data=['total'],              #values are extra data to be used in Dash callbacks
        labels={"Col1": "Feeling"},  # map the labels
        #title="Users' feelings",  # figure title
        template='plotly_dark',  # 'ggplot2', 'seaborn', 'simple_white', 'plotly',
        # 'plotly_white', 'plotly_dark', 'presentation',
        # 'xgridoff', 'ygridoff', 'gridon', 'none'
        width=800,  # figure width in pixels
        height=600,  # figure height in pixels
        hole=0.5,  # represents the hole in middle of pie
    )
    return pie_chart


if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=True)
