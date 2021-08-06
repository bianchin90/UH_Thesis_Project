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

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]

df = pd.read_excel("Georeferencing/sample_output.xlsx")


#function to get the dataframe updated
def getData():
    df_table = df
    #if n > 0:
    print('table')
    print(df_table)
    try:
        data = df_table.to_dict('records')
    except:
        data = df_table
    print(data)
    return data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

blackbold = {'color': 'black', 'font-weight': 'bold'}

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
    html.Div(children=[
        dcc.Graph(id='graph',
                  style={'display': 'inline-block', 'height': '100vh'},
                  config={'displayModeBar': False, 'scrollZoom': True},
                  # style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                  #        # , 'width':'180vh'
                  #        },
                  animate=True
                  ),
        dcc.Graph(id='pie-chart',
                  style={'display': 'inline-block',  'height': '100vh', 'width':'85vh', 'height':'45vh'},
                  # style={'padding-bottom': '2px', 'padding-left': '2px', 'height': '100vh'
                  #                        # , 'width':'80vh'
                  #                        },
                  animate=True),  # TEST WITH ANOTHER CHART
    ]),

    html.Div(children=[
        dash_table.DataTable(id='count-table',
                             columns=[{"name": i, "id": i} for i in df.columns],
                             data=getData(),
                             )
    ]),

    html.Br(),
    html.Br()
], className='ten columns offset-by-one' #you have a total of 12 columns
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
@app.callback(Output('pie-chart', 'figure'),
              Input('interval-component', 'n_intervals'))

def update_pie(n2):
    time = len(dff)
    dff.loc[len(dff)] = [time, rd.randint(1,101)]
    print(dff)

    # fig = px.line(dff, x=dff['X'], y=dff['Y'])
    # fig = fig.update_layout(yaxis={'title':'Count'}, xaxis={'title':'Timeline'},
    #                   title={'text':'Number of tweets detected',
    #                   'font':{'size':28},'x':0.5,'xanchor':'center'})
    data = go.Line(x=list(dff['X']), y=list(dff['Y']), mode= 'lines+markers')
    return {'data':data, 'layout' : go.Layout(xaxis=dict(range=[dff['X'].min(),dff['X'].max()], title='Timeline'),
                                                yaxis=dict(range=[dff['Y'].min(),dff['Y'].max()], title='Count' ))
            }


# -----------------------------------------------------------------
# test with table
@app.callback(Output('count-table', 'data'),
              Input('interval-component', 'n_intervals')
              # ,
              # [State('count-table', 'data')]
)

def update_table(n):
    return getData()
# #--------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=True)
