#------------------------------------------------------delete this section and add your own mapbox token below
# import yaml #(pip install pyyaml)
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
# mapbox_access_token = cfg['mysql']['key']
#-------------------------------------------------------


import pandas as pd
import numpy as np
import dash                     #(version 1.0.0)
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
import plotly.offline as py     #(version 4.4.1)
import plotly.graph_objs as go

with open('Profile.json') as profile:
    config = json.load(profile)

mapbox_access_token = config["Mapbox_Access_Token"]


df = pd.read_excel("Georeferencing/sample_output.xlsx")

app = dash.Dash(__name__)

blackbold={'color':'black', 'font-weight': 'bold'}

app.layout = html.Div([
#---------------------------------------------------------------
# Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
    html.Div([
        html.Div([
            # Map-legend
            #  html.Ul([
            #      html.Li("1 tweet", className='circle', style={'background': '#ff00ff','color':'black',
            #          'list-style':'none','text-indent': '17px'}),
            #      html.Li("2 tweets", className='circle', style={'background': '#0000ff','color':'black',
            #          'list-style':'none','text-indent': '17px','white-space':'nowrap'}),
            #     html.Li("Hazardous_waste", className='circle', style={'background': '#FF0000','color':'black',
            #         'list-style':'none','text-indent': '17px'}),
            #     html.Li("Plastic_bags", className='circle', style={'background': '#00ff00','color':'black',
            #         'list-style':'none','text-indent': '17px'}),
            #     html.Li("Recycling_bins", className='circle',  style={'background': '#824100','color':'black',
            #         'list-style':'none','text-indent': '17px'}),
            #  ], style={'border-bottom': 'solid 3px', 'border-color':'#00FC87','padding-top': '6px'}
            #  ),

            # Borough_checklist
            # html.Label(children=['Borough: '], style=blackbold),
            # dcc.Checklist(id='boro_name',
            #         options=[{'label':str(b),'value':b} for b in sorted(df['boro'].unique())],
            #         value=[b for b in sorted(df['boro'].unique())],
            # ),

            # tweet count checklist
            html.Label(children=['Looking to recycle: '], style=blackbold),
            dcc.Checklist(id='recycling_type',
                    options=[{'label':str(b),'value':b} for b in sorted(df['tweets'].unique())],
                    value=[b for b in sorted(df['tweets'].unique())],
            ),

            # Web_link
            # html.Br(),
            # html.Label(['Website:'],style=blackbold),
            # html.Pre(id='web_link', children=[],
            # style={'white-space': 'pre-wrap','word-break': 'break-all',
            #      'border': '1px solid black','text-align': 'center',
            #      'padding': '12px 12px 12px 12px', 'color':'blue',
            #      'margin-top': '3px'}
            # ),

        ], className='three columns'
        ),

        # Map
        html.Div([
            dcc.Graph(id='graph', config={'displayModeBar': False, 'scrollZoom': True},
                style={'padding-bottom':'2px','padding-left':'2px','height':'100vh'
                    #, 'width':'80vh'
                       }
            )
        ], className='nine columns'
        ),

    ], className='row'
    ),

], className='ten columns offset-by-one'
)

#---------------------------------------------------------------
# Output of Graph
@app.callback(Output('graph', 'figure'),
              #Input('boro_name', 'value'),
               Input('recycling_type', 'value')
              )

def update_figure(chosen_boro):
    df_sub = df[(df['tweets'].isin(chosen_boro))]

    # Create figure
    locations=[go.Scattermapbox(
                    lon = df_sub['lon'],
                    lat = df_sub['lat'],
                    mode='markers',
                    marker={'color' : df_sub['tweets']},
                   # unselected={'marker' : {'opacity':1}},
                    selected={'marker' : {'size':25}},
                    hoverinfo='text',
                    hovertext=df_sub['tweets'],
                    customdata=df_sub['city'],

    )]

    # Return figure
    return {
        'data': locations,
        'layout': go.Layout(
            uirevision= 'foo', #preserves state of figure/map after callback activated
            clickmode= 'event+select',
            hovermode='closest',
            hoverdistance=2,
            title=dict(text="Where is the earthquake?",font=dict(size=50, color='green')),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0, #orientation
                style='light',
                center=dict(
                    lat=42.44208797622657,
                    lon=12.966702481337714
                ),

                pitch=0, #incidence angle
                zoom=5
            ),
        )
    }
#---------------------------------------------------------------
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
    app.run_server(debug=False)