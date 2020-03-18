import argparse
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask
import plotly.graph_objs as go
import numpy as np


PARAMETER_TYPES = ('beta_1', 'beta_2')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)


if __name__ == '__main__':
    parser = argparse.ArgumentParse()
    parser.add_argument('--file', type=str, default=None,
                        help='JSON Line file of training data generated with W&B')
    args = parser.parse_args()

    random_x = np.random.randint(1, 21, 20)
    random_y = np.random.randint(1, 21, 20)

    app.layout = html.Div([
        dcc.Graph(
            id='scatter',
            figure={
                'data': [
                    go.Scatter(x=random_x, y=random_y, mode='markers')
                ],
                'layout': go.Layout(
                    title='Smoothing Visualizer',
                    xaxis={'title': 'X axis'},
                    yaxis={'title': 'Y axis'},
                )
            }
        ),
        dcc.Slider(id='beta_1', min=0, max=1, step=0.01, value=0.5),
        html.Div(id='slider_beta_1'),
        dcc.Slider(id='beta_2', min=0, max=1, step=0.01, value=0.5),
        html.Div(id='slider_beta_2'),
    ])

    @app.callback(Output('slider_beta_1', 'children'), [Input('beta_1', 'value')])
    def update_beta_1(value):
        return f'beta_1: {value}'

    @app.callback(Output('slider_beta_2', 'children'), [Input('beta_2', 'value')])
    def update_beta_2(value):
        return f'beta_2: {value}'

    app.run_server()
