import base64

import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
# import twitter  # pip install python-twitter
from app import app
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



# Connect to the layout and callbacks of each tab
# from mentions import mentions_layout
# from trends import trends_layout
# from other import other_layout
from summary import summary_layout
from map import map_layout


def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:images/png;base64,' + base64.b64encode(image).decode('utf-8')

# our app's Tabs *********************************************************
app_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Summary", tab_id="tab-summary", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
                dbc.Tab(label="Map", tab_id="tab-map", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
                dbc.Tab(label="Trend", tab_id="tab-other", labelClassName="text-success font-weight-bold", activeLabelClassName="text-danger"),
            ],
            id="tabs",
            active_tab="tab-summary",
        ),
    ], className="mt-3"
)

app.layout = dbc.Container([
    dcc.Interval(
            id='interval-component',
            interval=1 * 10000,  # in milliseconds
            n_intervals=0
        ),
    dbc.Row(children=[
        html.Br(),
        dbc.Row([
            html.Img(src=b64_image(r"C:\Users\filip\PycharmProjects\UH_Thesis_Project\images\logo2.png"), style={"margin-right": "10vh"}),
            html.H4("Argo", className="card-title",
                    style={'font-size': 64, 'text-align': 'center', 'color': "black"}),
            # color app_settings['div_text']
            html.Img(src=b64_image(r"C:\Users\filip\PycharmProjects\UH_Thesis_Project\images\logo2.png"), style={"margin-left": "10vh"}),
        ], className="mb-4", style={"margin-left": "27%"}),
        # html.Br()
    ], style={'backgroundColor': '#00b300'}),
    # dbc.Row(dbc.Col(html.H1("Twitter Analytics Dashboard Live",
    #                         style={"textAlign": "center"}), width=12)),
    # html.Hr(),
    dbc.Row(dbc.Col(app_tabs, width=12), className="mb-3"),
    html.Div(id='content', children=[])
])


@app.callback(
    Output("content", "children"),
    [Input("tabs", "active_tab")]
)
def switch_tab(tab_chosen):
    if tab_chosen == "tab-summary":
        print('hello')
        return summary_layout
    elif tab_chosen == "tab-map":
        print('hello 2')
        return map_layout
    elif tab_chosen == "tab-other":
        # return other_layout
        print('hello 3')
    return html.P("This shouldn't be displayed for now...")


if __name__=='__main__':
    app.run_server(debug=True)