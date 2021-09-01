import dash
import dash_bootstrap_components as dbc
# import twitter


# ensure you connected correctly
# print(api.VerifyCredentials())
# exit()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )