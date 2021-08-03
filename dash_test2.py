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
from dash.dependencies import Input, Output
import json
import plotly
import plotly.offline as py  # (version 4.4.1)
import plotly.graph_objs as go

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]

df = pd.read_excel("Georeferencing/sample_output.xlsx")

app = dash.Dash(__name__)

blackbold = {'color': 'black', 'font-weight': 'bold'}


app.layout = html.Div([
    # ---------------------------------------------------------------
    # Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
    html.Div([
        html.Div([
        ], className='three columns'
        ),

        # Map
        html.Div([
            dcc.Graph(id='graph', config={'displayModeBar': False, 'scrollZoom': True},
                      style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                              #, 'width':'180vh'
                             },
                      animate=True
                      ),
            dcc.Graph(id='pie-chart',style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                              #, 'width':'80vh'
                             },
                      animate=True), # TEST WITH ANOTHER CHART
            dcc.Interval(
                id='interval-component',
                interval=1 * 10000, # in milliseconds
                n_intervals=0
            )
        ], className='nine columns'
        ),

    ], className='row'
    ),

], className='ten columns offset-by-one'
)

# ---------------------------------------------------------------
# Output of Graph
@app.callback(Output('graph', 'figure'),
              [Input('interval-component', 'n_intervals')]
              )
def update_figure(n):
    # df_sub = df[(df['tweets'].isin(chosen_boro))]
    df_sub = df

    print(df_sub)
    if n == 0:
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
        marker={'color': df_sub['tweets'], 'size':10},
        # marker={'color': 'yellow', 'size':10},
        #unselected={'marker' : {'opacity':0, 'color' : 'black'}},
        #selected={'marker': {'size': 5}},
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
# test with new chart
@app.callback(Output('pie-chart', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_pie(n):

    traces = list()
    for t in range(2):
        traces.append(plotly.graph_objs.Bar(
            x=[1, 2, 3, 4, 5],
            y=[(t + 1) * random() for i in range(5)],
            name='Bar {}'.format(t)
            ))
    layout = go.Layout(
        title=dict(text="How is feeling the population?", font=dict(size=50, color='green')),
        barmode='group'
)
    return {'data': traces, 'layout': layout}



# ---------------------------------------------------------------
# callback for Web_link
# @app.callback(
#     Output('web_link', 'children'),
#     [Input('graph', 'clickData')])
# def display_click_data(clickData):
#     if clickData is None:
#         return 'Click on any bubble'
#     else:
#         # print (clickData)
#         the_link=clickData['points'][0]['customdata']
#         if the_link is None:
#             return 'No Website Available'
#         else:
#             return html.A(the_link, href=the_link, target="_blank")
# #--------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=True)
