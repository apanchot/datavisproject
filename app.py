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

summable_indicators = ['arrivals', 'income']

#places= ['energy_emissions', 'industry_emissions',
#       'agriculture_emissions', 'waste_emissions',
#       'land_use_foresty_emissions', 'bunker_fuels_emissions',
#       'electricity_heat_emissions', 'construction_emissions',
#       'transports_emissions', 'other_fuels_emissions']

######################################################Interactive Components############################################

city_options = [dict(label=city, value=city) for city in selected_cities.index]

indicator_options = [dict(label=indicator, value=indicator) for indicator in indicator_names]

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

                html.Div([html.Label(id='indic_1')], className='mini pretty'),
                html.Div([html.Label(id='indic_2')], className='mini pretty'),
            ], className='5 containers row'),

            html.Div([dcc.Graph(id='bar_graph')], className='bar_plot pretty')

        ], className='column2')

    ], className='row'),

    html.Div([

        html.Div([dcc.Graph(id='scattergeo')], className='column3 pretty')

    ], className='row')

])

######################################################Callbacks#########################################################

@app.callback(
    [
        Output("bar_graph", "figure"),
        Output("scattergeo", "figure")    ],
    [
        Input("button", 'n_clicks')
    ],
    [
        State("city_drop", "value"),
        State("indicator", "value"),
        State("lin_log", "value"),
        State("projection", "value")    ]
)
def plots(n_clicks, cities, indicator, scale, projection):

    ############################################First Bar Plot##########################################################
    data_bar = []

    x_bar = selected_cities.index
    y_bar = selected_cities[indicator]

    data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=indicator))

    layout_bar = dict(title=dict(text='Indicator per City'),
                  yaxis=dict(title='Indicator Value', type=['linear', 'log'][scale]),
                  paper_bgcolor='#f9f9f9'
                  )

    #############################################Second ScatterGeo######################################################

    map_data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == 'Generation_0',"x_coordinate"] , 
                     lon=df.loc[df.loc[:,"generation"] == 'Generation_0',"y_coordinate"] ,
                     hoverinfo = 'text',
                     text = df.loc[df.loc[:,"generation"] == 'Generation_0',"name_city"],
                     mode="lines+markers",
                     line=dict(width=1, color="blue"),
                     marker=dict(size=4, color="red"))]
    
    map_layout=go.Layout(
        title_text="Optimized World Tour", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None]),
                                   dict(label="Pause",
                                        method="animate",
                                        args=[None])])])
    
    map_frames=[go.Frame(
        data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == k,"x_coordinate"] , 
                     lon=df.loc[df.loc[:,"generation"] == k,"y_coordinate"] ,
                     text = df.loc[df.loc[:,"generation"] == k,"name_city"],
                     mode="lines+markers",
                     line=dict(width=((df.loc[df.loc[:,"generation"] == k,"norm_distance"].iloc[0])+0.1)*8, color="blue"),
                     marker=dict(size=4, color="red"))])

        for k in df.loc[:,"generation"].unique()]

    return go.Figure(data=data_bar, layout=layout_bar), \
           go.Figure(data=map_data, layout=map_layout, frames=map_frames)

@app.callback(
    [
        Output("indic_1", "children"),
        Output("indic_2", "children"),
    ],

    [
        Input("city_drop", "value")
    ]
)
def indicator(cities):
    cities_sum = selected_cities.loc[selected_cities.index.isin(cities)].sum()

    value_1 = cities_sum[summable_indicators[0]]
    value_2 = cities_sum[summable_indicators[1]]
    
    return str(summable_indicators[0]) + ': ' + str(value_1) + 'people',\
           str(summable_indicators[1]) + ': $' + str(value_2) + 'Billion',

if __name__ == '__main__':
    app.run_server(debug=True)