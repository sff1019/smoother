import argparse
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask
import jsonlines
import plotly.graph_objs as go
import numpy as np


PARAMETER_TYPES = ('beta_1', 'beta_2')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)


def exponential_smoothing(beta_1, beta_2, train_loss):
    ses_loss, des_loss = [], []
    b = 0

    for epoch in range(len(train_loss)):
        if epoch == 0:
            ses_loss.append(train_loss[epoch])
            des_loss.append(train_loss[epoch])
        elif epoch == 1:
            b = train_loss[epoch] - des_loss[-1]
            des_loss.append(
                (1 - beta_1) * (des_loss[-1] + b) + beta_1 * train_loss[epoch])
            ses_loss.append(
                (1 - beta_1) * ses_loss[-1] + beta_1 * train_loss[epoch])
        else:
            des_loss.append(
                (1 - beta_1) * (des_loss[-1] + b) + beta_1 * train_loss[epoch])
            b = (1 - beta_2) * b + beta_2 * \
                (des_loss[-1] - des_loss[-2])
            ses_loss.append(
                (1 - beta_1) * ses_loss[-1] + beta_1 * train_loss[epoch])

    return ses_loss, des_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None,
                        help='JSON Line file of training data generated with W&B')
    args = parser.parse_args()

    epochs = []
    train_loss, ses_loss, des_loss = [], [], []
    beta_1 = 0.7
    beta_2 = 0.6

    with jsonlines.open(args.file) as r:
        for obj in r:
            if 'ses_loss' in obj:
                epochs.append(obj['epoch'])
                train_loss.append(obj['train_loss'])
    ses_loss, des_loss = exponential_smoothing(beta_1, beta_2, train_loss)

    fig = go.Figure(
        layout=go.Layout(
            title='Smoothing Visualizer',
            xaxis={'title': 'Epoch'},
            yaxis={'title': 'Loss'},
        )
    )
    fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                             mode='lines', name='train loss'))
    fig.add_trace(go.Scatter(x=epochs, y=ses_loss,
                             mode='lines', name='ses loss'))
    fig.add_trace(go.Scatter(x=epochs, y=des_loss,
                             mode='lines', name='des loss'))

    app.layout = html.Div([
        dcc.Graph(id='plot', figure=fig),
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

    @app.callback(
        Output('plot', 'figure'),
        [Input('beta_1', 'value'), Input('beta_2', 'value')]
    )
    def update_exponential_smoothing(beta_1, beta_2):
        ses_loss, des_loss = exponential_smoothing(beta_1, beta_2, train_loss)
        fig = go.Figure(
            layout=go.Layout(
                title='Smoothing Visualizer',
                xaxis={'title': 'Epoch'},
                yaxis={'title': 'Loss'},
            )
        )
        fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                                 mode='lines', name='train loss'))
        fig.add_trace(go.Scatter(x=epochs, y=ses_loss,
                                 mode='lines', name='ses loss'))
        fig.add_trace(go.Scatter(x=epochs, y=des_loss,
                                 mode='lines', name='des loss'))
        return fig

    app.run_server(port='5000')
