import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import plotly.graph_objs as go


######################################################Data##############################################################

df = pd.read_csv('data/emission_full.csv')

gas_names = ['CO2_emissions', 'GHG_emissions', 'CH4_emissions','N2O_emissions', 'F_Gas_emissions']

places= ['energy_emissions', 'industry_emissions',
       'agriculture_emissions', 'waste_emissions',
       'land_use_foresty_emissions', 'bunker_fuels_emissions',
       'electricity_heat_emissions', 'construction_emissions',
       'transports_emissions', 'other_fuels_emissions']

######################################################Interactive Components############################################

country_options = [dict(label=country, value=country) for country in df['country_name'].unique()]

gas_options = [dict(label=gas.replace('_', ' '), value=gas) for gas in gas_names]

sector_options = [dict(label=place.replace('_', ' '), value=place) for place in places]


app = dash.Dash(__name__)

app.layout = html.Div([

    html.Label('Country Choice'),
    dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Portugal'],
        multi=True
    ),

    html.Br(),

    html.Label('Gas Choice'),
    dcc.Dropdown(
        id='gas_option',
        options=gas_options,
        value='CO2_emissions',
    ),

    html.Br(),

    html.Label('Sector Choice'),
    dcc.Dropdown(
        id='sector_options',
        options=sector_options,
        value=['energy_emissions', 'waste_emissions'],
        multi=True
    ),

    html.Br(),

    html.Label('Year Slider'),
    dcc.Slider(
        id='year_slider',
        min=df['year'].min(),
        max=df['year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in [1990, 1995, 2000, 2005, 2010, 2014]},
        value=df['year'].min(),
        step=1
    ),

    html.Br(),

    html.Label('Linear Log'),
    dcc.RadioItems(
        id='lin_log',
        options=[dict(label='Linear', value=0), dict(label='log', value=1)],
        value=0
    ),

    html.Br(),

    html.Label('Projection'),
    dcc.RadioItems(
        id='projection',
        options=[dict(label='Equirectangular', value=0), dict(label='Orthographic', value=1)],
        value=0
    ),

    html.Br(),

    dcc.Graph(id='bar_graph'),

    html.Br(),

    dcc.Graph(id='choropleth'),

    html.Br(),

    dcc.Graph(id='aggregate_graph'),

    html.Br(),

    html.Label(id='gas_1'),

    html.Br(),

    html.Label(id='gas_2'),

    html.Br(),

    html.Label(id='gas_3'),

    html.Br(),

    html.Label(id='gas_4'),

    html.Br(),

    html.Label(id='gas_5'),
])



######################################################Callbacks#########################################################


@app.callback(
    [
        Output("bar_graph", "figure"),
        Output("choropleth", "figure"),
        Output("aggregate_graph", "figure"),
    ],
    [
        Input("year_slider", "value"),
        Input("country_drop", "value"),
        Input("gas_option", "value"),
        Input("lin_log", "value"),
        Input("projection", "value"),
        Input("sector_options", "value")
    ]
)
def plots(year, countries, gas, scale, projection, sector):

    ############################################First Bar Plot##########################################################
    data_bar = []
    for country in countries:
        df_bar = df.loc[(df['country_name'] == country)]

        x_bar = df_bar['year']
        y_bar = df_bar[gas]

        data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=country))

    layout_bar = dict(title=dict(text='Emissions from 1990 until 2015'),
                  yaxis=dict(title='Emissions', type=['linear', 'log'][scale]),
                  paper_bgcolor='#f9f9f9'
                  )

    #############################################Second Choropleth######################################################

    df_emission_0 = df.loc[df['year'] == year]

    z = np.log(df_emission_0[gas])

    data_choropleth = dict(type='choropleth',
                           locations=df_emission_0['country_name'],
                           # There are three ways to 'merge' your data with the data pre embedded in the map
                           locationmode='country names',
                           z=z,
                           text=df_emission_0['country_name'],
                           colorscale='inferno',
                           colorbar=dict(title=str(gas.replace('_', ' ')) + ' (log scaled)'),

                           hovertemplate='Country: %{text} <br>' + str(gas.replace('_', ' ')) + ': %{z}',
                           name=''
                           )

    layout_choropleth = dict(geo=dict(scope='world',  # default
                                      projection=dict(type=['equirectangular', 'orthographic'][projection]
                                                      ),
                                      # showland=True,   # default = True
                                      landcolor='black',
                                      lakecolor='white',
                                      showocean=True,  # default = False
                                      oceancolor='azure',
                                      bgcolor='#f9f9f9'
                                      ),

                             title=dict(text='World ' + str(gas.replace('_', ' ')) + ' Choropleth Map on the year ' + str(year),
                                        x=.5  # Title relative position according to the xaxis, range (0,1)

                                        ),
                             paper_bgcolor='#f9f9f9'
                             )

    ############################################Third Scatter Plot######################################################

    df_loc = df.loc[df['country_name'].isin(countries)].groupby('year').sum().reset_index()

    data_agg = []

    for place in sector:
        data_agg.append(dict(type='scatter',
                         x=df_loc['year'].unique(),
                         y=df_loc[place],
                         name=place.replace('_', ' '),
                         mode='markers'
                         )
                    )

    layout_agg = dict(title=dict(text='Aggregate CO2 Emissions by Sector'),
                     yaxis=dict(title=['CO2 Emissions', 'CO2 Emissions (log scaled)'][scale],
                                type=['linear', 'log'][scale]),
                     xaxis=dict(title='Year'),
                     paper_bgcolor='#f9f9f9'
                     )

    return go.Figure(data=data_bar, layout=layout_bar), \
           go.Figure(data=data_choropleth, layout=layout_choropleth),\
           go.Figure(data=data_agg, layout=layout_agg)


@app.callback(
    [
        Output("gas_1", "children"),
        Output("gas_2", "children"),
        Output("gas_3", "children"),
        Output("gas_4", "children"),
        Output("gas_5", "children")
    ],
    [
        Input("country_drop", "value"),
        Input("year_slider", "value"),
    ]
)
def indicator(countries, year):
    df_loc = df.loc[df['country_name'].isin(countries)].groupby('year').sum().reset_index()

    value_1 = round(df_loc.loc[df_loc['year'] == year][gas_names[0]].values[0], 2)
    value_2 = round(df_loc.loc[df_loc['year'] == year][gas_names[1]].values[0], 2)
    value_3 = round(df_loc.loc[df_loc['year'] == year][gas_names[2]].values[0], 2)
    value_4 = round(df_loc.loc[df_loc['year'] == year][gas_names[3]].values[0], 2)
    value_5 = round(df_loc.loc[df_loc['year'] == year][gas_names[4]].values[0], 2)

    return str(gas_names[0]).replace('_', ' ') + ': ' + str(value_1),\
           str(gas_names[1]).replace('_', ' ') + ': ' + str(value_2), \
           str(gas_names[2]).replace('_', ' ') + ': ' + str(value_3), \
           str(gas_names[3]).replace('_', ' ') + ': ' + str(value_4), \
           str(gas_names[4]).replace('_', ' ') + ': ' + str(value_5),


if __name__ == '__main__':
    app.run_server(debug=True)