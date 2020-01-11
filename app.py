import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from math import sin, cos, sqrt, atan2, radians
from ga import (
    initial,
    fitness_aux,
    fitness_function,
    tournament_selection,
    select_parents,
    order_crossover,
    inversion_mutation,
    elitism_replacement,
    save_best_fitness
)

#################### Importing all the needed data ####################

#Importing a dataframe that contains latitude and longitude coordinates of 15,493 cities from around the world.
cities_coordinates = pd.read_csv('./data/worldcities.csv')

#Importing a dataframe that contains tourism ranking and arrivals data
cities_visitors = pd.read_csv('./data/wiki_international_visitors.csv')

#################### Function to calculate the distance between cities ####################

def distance(x, y):
    R = 6373.0
    
    lat1 = radians(selected_cities.loc[x,'lat'])
    lon1 = radians(selected_cities.loc[x,'lng'])
    lat2 = radians(selected_cities.loc[y,'lat'])
    lon2 = radians(selected_cities.loc[y,'lng'])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance

#################### Selecting some cities ####################

selected = ['Tokyo','Miami','Lima','Rio de Janeiro','Los Angeles','Buenos Aires','Rome','Lisbon','Paris',
            'Munich','Delhi','Sydney','Moscow','Istanbul','Johannesburg','Madrid','Seoul','London','Bangkok',
            'Toronto','Dubai','Beijing', 'Abu Dhabi', 'Stockholm']
selected_cities = cities_coordinates.loc[cities_coordinates['city'].isin(selected),['city','country','lat','lng']]
selected_cities.drop_duplicates(subset='city', keep='first', inplace=True)
selected_cities.reset_index(inplace = True, drop = True)
selected_cities = selected_cities.drop('country', axis = 1)
selected_cities.set_index('city', inplace = True)
cities_visitors.set_index('City', inplace = True)
selected_cities = selected_cities.merge(cities_visitors[['Rank(Euromonitor)',
                                                   'Arrivals 2018(Euromonitor)',
                                                   'Growthin arrivals(Euromonitor)',
                                                   'Income(billions $)(Mastercard)']], left_index=True, right_index=True, how='left')

selected_cities.rename(columns={'Rank(Euromonitor)':'rank',
                                'Arrivals 2018(Euromonitor)':'arrivals',
                                'Growthin arrivals(Euromonitor)':'growth',
                                'Income(billions $)(Mastercard)':'income'}, inplace=True)

#Calculating the distance between them
data = [[distance(i,j) for j in selected_cities.index] for i in selected_cities.index]

#################### Running the Genetic Algorithm ####################

decision_variables = list(range(len(data)))
population = initial(decision_variables, 20)
fitness = fitness_function(population, data)
best = save_best_fitness(population, fitness)
generation, best_fitness, fittest = [0], [best[1]], [str(best[0])]

for gen in range(1000):
    parents = select_parents(population, fitness)
    offspring = parents.copy()
    for i in range(0,len(population),2):
        if (np.random.uniform() < 0.6):
            offspring[i],offspring[i+1] = order_crossover(parents[i],parents[i+1])
    for i in range(len(population)):
        if (np.random.uniform() < 0.6):
            offspring[i] = inversion_mutation(offspring[i])
    fitness_offspring = fitness_function(offspring, data)
    population = elitism_replacement(population, fitness, offspring, fitness_offspring)
    fitness = fitness_function(population, data)
    best = save_best_fitness(population, fitness)
    generation.append(gen+1), best_fitness.append(best[1]), fittest.append(str(best[0]))

generation = pd.Series(generation)
best_fitness = pd.Series(best_fitness)
fittest = pd.Series(fittest)
run = pd.concat([generation, best_fitness, fittest], axis = 1)
run.columns = ['Generation', 'Fitness', 'Fittest']
run.drop_duplicates('Fittest', inplace=True)

#################### Preparing the GA results dataframe ####################

#Function to return the cities-path with the best fitness (lowest distance)
def path(x):
    best_fitness_aux = run.loc[x,'Fittest'].replace(',','').replace('[','').replace(']','').split(' ')
    path_best_fitness = [int(i) for i in best_fitness_aux]
    path_best_fitness = path_best_fitness + [path_best_fitness[0]]
    return path_best_fitness

generation = lambda x: ['Generation_'+str(run.loc[x,'Generation'])]*len(path(x))
total_distance = lambda x: [run.loc[x,'Fitness']]*len(path(x))

all_path = []
all_generation = []
all_distances = []
for i in run.loc[:,'Generation']:
    all_path = all_path + path(i)
    all_generation = all_generation + generation(i) 
    all_distances = all_distances + total_distance(i)

all_generation = pd.Series(all_generation)
all_path = pd.Series(all_path)
all_distances = pd.Series(all_distances)

x_coordinate = [selected_cities.iloc[i,0] for i in all_path]
y_coordinate = [selected_cities.iloc[i,1] for i in all_path]
name_city = [selected_cities.index[i] for i in all_path]
x_coordinate = pd.Series(x_coordinate)
y_coordinate = pd.Series(y_coordinate)
name_city = pd.Series(name_city)

#Create a dataframe with TSP problem GA results, cities names and coordinates
df = pd.concat([all_generation, all_path, all_distances, name_city, x_coordinate, y_coordinate], axis = 1)
df.columns = ['generation', 'city', 'distance', 'name_city', 'x_coordinate','y_coordinate']

#Insert a column with the normalized distance (to be used as line width in the graph)
df['norm_distance'] = ''
max_ = df['distance'].max()
min_ = df['distance'].min()
for idx in df.index:
    df.at[idx, 'norm_distance'] = (df['distance'].loc[idx] - min_)/(max_ - min_)

######################################################Data##############################################################

indicator_names = ['rank', 'arrivals', 'growth', 'income']

#places= ['energy_emissions', 'industry_emissions',
#       'agriculture_emissions', 'waste_emissions',
#       'land_use_foresty_emissions', 'bunker_fuels_emissions',
#       'electricity_heat_emissions', 'construction_emissions',
#       'transports_emissions', 'other_fuels_emissions']

######################################################Interactive Components############################################

city_options = [dict(label=city, value=city) for city in df['city'].unique()]

indicator_options = [dict(label=indicator.replace('_', ' '), value=indicator) for indicator in indicator_names]

#sector_options = [dict(label=place.replace('_', ' '), value=place) for place in places]

##################################################APP###############################################################

app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.H1('World Tour Simulator')
    ], className='Title'),

    html.Div([

        html.Div([
            dcc.Tabs(id="tabs", value='tab_1', children=[
                dcc.Tab(label='Tab_1', value='tab_1', children=[
                                                                    html.Label('Cities'),
                                                                    dcc.Dropdown(
                                                                        id='city_drop',
                                                                        options=city_options,
                                                                        value=['Lisbon','Rio de Janeiro'],
                                                                        multi=True
                                                                    ),

                                                                    html.Br(),

                                                                    html.Label('Tourism Indicator'),
                                                                    dcc.Dropdown(
                                                                        id='indicator',
                                                                        options=indicator_options,
                                                                        value='arrivals',
                                                                    ),

                                                                    html.Br(),
                ]),
                dcc.Tab(label='Tab_2',value='tab_2', children=[
                                                            html.Label('City Slider'),
                                                            dcc.Slider(
                                                                id='city_slider',
                                                                min=selected_cities.sort_values('city').iloc[0].name,
                                                                max=selected_cities.sort_values('city').iloc[-1].name,
                                                                marks={str(i): '{}'.format(str(i)) for i in list(selected_cities.sort_values('city').index.values)},
                                                                value=selected_cities.sort_values('city').iloc[0].name,
                                                                step=None,
                                                                included=False
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
                                                            )
                ]),
            ]),
            html.Button('Submit', id="button")

        ], className='column1 pretty'),

        html.Div([

            html.Div([

                html.Div([html.Label(id='gas_1')], className='mini pretty'),
                html.Div([html.Label(id='gas_2')], className='mini pretty'),
                html.Div([html.Label(id='gas_3')], className='mini pretty'),
                html.Div([html.Label(id='gas_4')], className='mini pretty'),
                html.Div([html.Label(id='gas_5')], className='mini pretty'),

            ], className='5 containers row'),

            html.Div([dcc.Graph(id='bar_graph')], className='bar_plot pretty')

        ], className='column2')

    ], className='row'),

    html.Div([

        html.Div([dcc.Graph(id='choropleth')], className='column3 pretty'),

        html.Div([dcc.Graph(id='aggregate_graph')], className='column3 pretty')

    ], className='row')

])

######################################################Callbacks#########################################################

@app.callback(
    [
        Output("bar_graph", "figure"),
        Output("choropleth", "figure"),
        Output("aggregate_graph", "figure"),
    ],
    [
        Input("button", 'n_clicks')
    ],
    [
        State("city_slider", "value"),
        State("city_drop", "value"),
        State("indicator", "value"),
        State("lin_log", "value"),
        State("projection", "value")    ]
)
def plots(n_clicks,year, indicators, gas, scale, projection, sector):

    ############################################First Bar Plot##########################################################
    data_bar = []
    for indicator in indicators:
        df_bar = selected_cities.copy()

        x_bar = df_bar['city']
        y_bar = df_bar[gas]

        data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=indicator))

    layout_bar = dict(title=dict(text='Indicator per City'),
                  yaxis=dict(title='Indicator Value', type=['linear', 'log'][scale]),
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

    df_loc = df.loc[df['country_name'].isin(indicators)].groupby('year').sum().reset_index()

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
def indicator(indicators, year):
    df_loc = df.loc[df['country_name'].isin(indicators)].groupby('year').sum().reset_index()

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