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
    ses_list, des_list = [], []
    b = 0.

    prev_ses_error = train_loss[0]
    prev_des_error = train_loss[0]
    for epoch in range(len(train_loss)):
        if epoch == 0:
            ses_list.append(prev_ses_error)
            des_list.append(prev_des_error)
        elif epoch == 1:
            b = train_loss[epoch] - prev_des_error
            des_error = (1 - beta_1) * (prev_des_error + b) + \
                beta_1 * train_loss[epoch]
            ses_error = (1 - beta_1) * prev_ses_error + \
                beta_1 * train_loss[epoch]
            des_list.append(des_error)
            ses_list.append(ses_error)
            prev_ses_error, prev_des_error = ses_error, des_error
        else:
            des_error = (1 - beta_1) * (prev_des_error + b) + \
                beta_1 * train_loss[epoch]
            ses_error = (1 - beta_1) * prev_ses_error + \
                beta_1 * train_loss[epoch]
            des_list.append(des_error)
            ses_list.append(ses_error)
            b = (1 - beta_2) * b + beta_2 * (des_error - prev_des_error)
            prev_ses_error, prev_des_error = ses_error, des_error

    return ses_list, des_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None,
                        help='JSON Line file of training data generated with W&B')
    args = parser.parse_args()

    epochs = []
    train_loss, ses_error, des_error = [], [], []
    beta_1 = 0.2
    beta_2 = 0.3

    with jsonlines.open(args.file) as r:
        for obj in r:
            if 'ses_error' in obj:
                epochs.append(obj['global_step'])
                train_loss.append(obj['train_error'])
    ses_error, des_error = exponential_smoothing(beta_1, beta_2, train_loss)

    fig = go.Figure(
        layout=go.Layout(
            title='Smoothing Visualizer',
            xaxis={'title': 'Epoch'},
            yaxis={'title': 'Loss'},
        )
    )
    fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                             mode='lines', name='train loss'))
    fig.add_trace(go.Scatter(x=epochs, y=ses_error,
                             mode='lines', name='ses loss'))
    fig.add_trace(go.Scatter(x=epochs, y=des_error,
                             mode='lines', name='des loss'))

    app.layout = html.Div([
        dcc.Graph(id='plot', figure=fig),
        dcc.Slider(id='beta_1', min=0, max=1, step=0.01, value=0.2),
        html.Div(id='slider_beta_1'),
        dcc.Slider(id='beta_2', min=0, max=1, step=0.01, value=0.3),
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
        ses_error, des_error = exponential_smoothing(
            beta_1, beta_2, train_loss)
        fig = go.Figure(
            layout=go.Layout(
                title='Smoothing Visualizer',
                xaxis={'title': 'Epoch'},
                yaxis={'title': 'Loss'},
            )
        )
        fig.add_trace(go.Scatter(x=epochs, y=train_loss,
                                 mode='lines', name='train loss'))
        fig.add_trace(go.Scatter(x=epochs, y=ses_error,
                                 mode='lines', name='ses loss'))
        fig.add_trace(go.Scatter(x=epochs, y=des_error,
                                 mode='lines', name='des loss'))
        return fig

    app.run_server(port='5000')
