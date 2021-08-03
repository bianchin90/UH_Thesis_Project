# ------------------------------------------------------delete this section and add your own mapbox token below
# import yaml #(pip install pyyaml)
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
# mapbox_access_token = cfg['mysql']['key']
# -------------------------------------------------------
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
            dcc.Graph(id='pie-chart', animate=True), # TEST WITH ANOTHER CHART
            dcc.Interval(
                id='interval-component',
                interval=1 * 10000
            )
        ], className='nine columns'
        ),

    ], className='row'
    ),

], className='ten columns offset-by-one'
)


# -----------------------------------------------------------------
# test with new chart
@app.callback(Output('pie-chart', 'figure'),
              Input('interval-component', 'interval'))
def update_pie(input_data):

    traces = list()
    for t in range(2):
        traces.append(plotly.graph_objs.Bar(
            x=[1, 2, 3, 4, 5],
            y=[(t + 1) * random() for i in range(5)],
            name='Bar {}'.format(t)
            ))
    layout = go.Layout(
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
