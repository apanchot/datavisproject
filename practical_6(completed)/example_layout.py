import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
import dash
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State


df = pd.read_csv('Data/emission_full.csv')
partial = df[(df['year']==2000)&(df['country_name']=='Portugal')]

fig=go.Figure(data=go.Choropleth(locations=['Portugal'],
                                 z=[100], showscale=False, locationmode='country names'))

fig2 = go.Figure(data=go.Scatter(x=np.random.normal(0,1,100),
                                 y=np.random.normal(0,1,100), mode='lines'))


app = dash.Dash()

app.layout=html.Div([

    html.Div([
        html.H1('My Application Title'),
        html.H3('My subtitle')
    ], id='title_division', style={'border-style':'solid', 'border-color':'black'}),

    html.Div([
        dcc.Dropdown(
            options=[
                {'label': i, 'value': i} for i in df['country_name'].unique()
            ],
            multi=True,
            value="Portugal"
        )
    ], id='Multi-Dropdown'),


    html.Div([
        dcc.Graph(id='country_display', figure=fig)
    ],id='graph-division'),


    html.Div([
        dcc.Slider(
            min=df['year'].min(),
            max=df['year'].max(),
            #marks={str(i): '{}'.format(i) for i in df['year'].unique()},
            value=2000,
        )
    ],id='year-slider'),

    html.P(html.Div([
        dcc.Graph(id='line_plot', figure=fig2)
    ]))
], id='outer_division')


if __name__ == '__main__':
    app.run_server(debug=True)


