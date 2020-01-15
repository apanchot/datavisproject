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
    save_best_fitness,
    ga_search
)

#################### Importing all the needed data ####################

#Importing a dataframe that contains latitude and longitude coordinates of 15,493 cities from around the world.
cities_coordinates = pd.read_csv('./data/worldcities.csv')

#Importing a dataframe that contains tourism ranking and arrivals data
cities_visitors = pd.read_csv('./data/wiki_international_visitors.csv')

#Importing a dataframe with average hotel prices by city
hotel_prices = pd.read_excel('./data/average_hotel_prices.xlsx')

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
hotel_prices.set_index('city', inplace=True)
selected_cities = selected_cities.merge(cities_visitors[['Rank(Euromonitor)',
                                                   'Arrivals 2018(Euromonitor)',
                                                   'Growthin arrivals(Euromonitor)',
                                                   'Income(billions $)(Mastercard)']], left_index=True, right_index=True, how='left')

selected_cities = selected_cities.merge(hotel_prices[['hotel_price']], left_index=True, right_index=True, how='left')

selected_cities.rename(columns={'Rank(Euromonitor)':'rank',
                                'Arrivals 2018(Euromonitor)':'arrivals',
                                'Growthin arrivals(Euromonitor)':'growth',
                                'Income(billions $)(Mastercard)':'income'}, inplace=True)

selected_cities['norm_rank'] = (selected_cities['rank'] - selected_cities['rank'].min()) / (selected_cities['rank'].max() - selected_cities['rank'].min())

######################################################Data##############################################################

indicator_names = ['rank', 'arrivals', 'growth', 'income']

summable_indicators = ['arrivals', 'income', 'hotel_price']

######################################################Interactive Components############################################

city_options = [dict(label=city, value=city) for city in selected_cities.index]

indicator_options = [dict(label=indicator, value=indicator) for indicator in indicator_names]

##################################################APP###############################################################

app = dash.Dash(__name__)

tabs_styles = {
    'height': '44px',
    'width' : '100px'
}

app.layout = html.Div([

    html.Div([
        html.H1('World Tour Simulator')
    ], className='Title'),

    html.Div([

        html.Div([
            dcc.Tabs(id="tabs", value='tab_1', children=[
                dcc.Tab(label='Command Panel', value='tab_1', children=[
                                                                    html.Label('Cities'),
                                                                    dcc.Dropdown(
                                                                        id='city_drop',
                                                                        options=city_options,
                                                                        value=list(np.random.choice(selected_cities.index, 10, replace=False)),
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

                                                                    html.Label('What\'s the minimum ranking?'),
                                                                    dcc.Slider(
                                                                    id='rank_slider',
                                                                    min=0,
                                                                    max=100,
                                                                    marks={str(i): '{}'.format(str(i)) for i in list(np.linspace(1,100,11, dtype = int))},
                                                                    value=100,
                                                                    step=None
                                                                    ),

                                                                    html.Br(),
                ]),
            ]),
            html.Button('Submit', id="button")

        ], className='column1 pretty'),

        html.Div([

            html.Div([html.Label('Cities above the minimum rank:'), html.Label(id='indic_0')], className='mini pretty'),        
            
            html.Div([

                html.Div([html.Label(id='indic_1')], className='mini pretty'),
                html.Div([html.Label(id='indic_2')], className='mini pretty'),
                html.Div([html.Label(id='indic_3')], className='mini pretty'),
            ], className='5 containers row'),

            html.Div([dcc.Graph(id='scattergeo')], className='column3 pretty')

        ], className='column2')

    ], className='row'),

    html.Div([

        html.Div([dcc.Graph(id='bar_graph')], className='bar_plot pretty')

    ], className='row')

])

######################################################Callbacks#########################################################

@app.callback(
    [
        Output("bar_graph", "figure"),
        Output("scattergeo", "figure"),
    ],
    [
        Input("button", 'n_clicks')
    ],
    [
        State("city_drop", "value"),
        State("indicator", "value"),
        State("rank_slider", "value"),
    ]
)
def plots(n_clicks, cities, indicator, rank):

    ############################################First Bar Plot##########################################################
    data_bar = []

    new_selection = selected_cities.loc[cities,:].sort_values(by=[indicator])
    new_selection = new_selection.loc[new_selection['rank'] <= rank]

    x_bar = new_selection.index
    y_bar = new_selection[indicator]

    data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=indicator))

    layout_bar = dict(title=dict(text=indicator.title() + ' per City'),
                  yaxis=dict(title=indicator.title() + ' Value'),
                  paper_bgcolor='#f9f9f9'
                  )

    #############################################Second ScatterGeo######################################################

    data = [[distance(i, j) for j in new_selection.index] for i in new_selection.index]

    run = ga_search(data)

    def path(x):
        best_fitness_aux = run.loc[x, 'Fittest'].replace(',', '').replace('[', '').replace(']', '').split(' ')
        path_best_fitness = [int(i) for i in best_fitness_aux]
        path_best_fitness = path_best_fitness + [path_best_fitness[0]]
        return path_best_fitness

    generation = lambda x: ['Generation_' + str(run.loc[x, 'Generation'])] * len(path(x))
    total_distance = lambda x: [run.loc[x, 'Fitness']] * len(path(x))

    all_path = []
    all_generation = []
    all_distances = []
    for i in run.loc[:, 'Generation']:
        all_path = all_path + path(i)
        all_generation = all_generation + generation(i)
        all_distances = all_distances + total_distance(i)

    all_generation = pd.Series(all_generation)
    all_path = pd.Series(all_path)
    all_distances = pd.Series(all_distances)

    x_coordinate = [new_selection.iloc[i, 0] for i in all_path]
    y_coordinate = [new_selection.iloc[i, 1] for i in all_path]
    name_city = [new_selection.index[i] for i in all_path]
    x_coordinate = pd.Series(x_coordinate)
    y_coordinate = pd.Series(y_coordinate)
    name_city = pd.Series(name_city)

    df = pd.concat([all_generation, all_path, all_distances, name_city, x_coordinate, y_coordinate], axis=1)
    df.columns = ['generation', 'city', 'distance', 'name_city', 'x_coordinate', 'y_coordinate']
 
    if (df['distance'].max() - df['distance'].min() != 0):
        df['norm_distance'] = (df['distance'] - df['distance'].min()) / (df['distance'].max() - df['distance'].min())
    else:
        df['norm_distance'] = 0

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
        Output("indic_0", "children"),
        Output("indic_1", "children"),
        Output("indic_2", "children"),
        Output("indic_3", "children"),
    ],

    [
        Input("city_drop", "value"),
        Input("rank_slider", "value"),
    ]
)
def indicator(cities, rank):
    cities = selected_cities.loc[selected_cities.index.isin(cities)]
    cities = cities.loc[cities['rank'] <= rank]
    cities_sum = cities.sum()
    cities_avg = cities.mean()

    value_1 = round(cities_sum[summable_indicators[0]]/1000000,0)
    value_2 = round(cities_sum[summable_indicators[1]],2)
    value_3 = round(cities_avg[summable_indicators[2]],2)

    cities_selection = [str(x) for x in cities.index]
    cities_selection.sort()
    cities_selection = ', '.join(cities_selection)

    return cities_selection,\
           str(summable_indicators[0]).title() + ' sum: ' + str(value_1) + ' million people',\
           str(summable_indicators[1]).title() + ' sum: $' + str(value_2) + ' billion',\
           str(summable_indicators[2]).title() + ' mean: $' + str(value_3),

if __name__ == '__main__':
    app.run_server(debug=True)