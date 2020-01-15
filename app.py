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

# Adding Continents
Asia = cities_visitors.loc[:,'Country'].isin(['China','Thailand','Macau','Singapore','United Arab Emirates',
                                              'Malaysia','Turkey','India','Japan','Taiwan','Saudi Arabia',
                                              'South Korea','Vietnam','Indonesia','Israel','Philippines',
                                              'Iran','Lebanon','Sri Lanka','Jordan'])
Africa = cities_visitors.loc[:,'Country'].isin(['Egypt','South Africa','Morocco','Ghana','Nigeria'])
Europe = cities_visitors.loc[:,'Country'].isin(['United Kingdom','France','Italy','Czech Republic','Netherlands',
                                                'Spain', 'Austria','Germany','Greece', 'Russia', 'Ireland','Belgium', 
                                                'Hungary', 'Portugal', 'Denmark','Poland','Sweden','Switzerland',
                                                'Romania', 'Bulgaria'])
Oceania = cities_visitors.loc[:,'Country'].isin(['Australia','New Zealand'])
North_America = cities_visitors.loc[:,'Country'].isin(['United States','Mexico','Canada'])
South_America = cities_visitors.loc[:,'Country'].isin(['Argentina','Peru','Brazil','Colombia','Uruguay','Ecuador'])
Central_America = cities_visitors.loc[:,'Country'].isin(['Dominican Republic'])
cities_visitors.loc[:,'Continent'] = cities_visitors.loc[:,'Country']
cities_visitors.loc[:,'Continent'].mask(Asia, 'Asia', inplace=True)
cities_visitors.loc[:,'Continent'].mask(Africa, 'Africa', inplace=True)
cities_visitors.loc[:,'Continent'].mask(Europe, 'Europe', inplace=True)
cities_visitors.loc[:,'Continent'].mask(Oceania, 'Oceania', inplace=True)
cities_visitors.loc[:,'Continent'].mask(North_America, 'North America', inplace=True)
cities_visitors.loc[:,'Continent'].mask(South_America, 'South America', inplace=True)
cities_visitors.loc[:,'Continent'].mask(Central_America, 'Central America', inplace=True)

#Importing a dataframe with average hotel prices by city
hotel_prices = pd.read_excel('./data/average_hotel_prices.xlsx')

#Importing a dataframe with public transportation costs
transportation = pd.read_excel('./data/transportation_costs.xlsx')

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
transportation.set_index('city', inplace=True)
selected_cities = selected_cities.merge(cities_visitors[['Rank(Euromonitor)',
                                                   'Arrivals 2018(Euromonitor)',
                                                   'Growthin arrivals(Euromonitor)',
                                                   'Income(billions $)(Mastercard)', 'Continent']], left_index=True, right_index=True, how='left')

selected_cities = selected_cities.merge(hotel_prices[['hotel_price']], left_index=True, right_index=True, how='left')

selected_cities = selected_cities.merge(transportation[['average_ticket_dollars']], left_index=True, right_index=True, how='left')

selected_cities.rename(columns={'Rank(Euromonitor)':'rank',
                                'Arrivals 2018(Euromonitor)':'arrivals',
                                'Growthin arrivals(Euromonitor)':'growth',
                                'Income(billions $)(Mastercard)':'income',
                                'average_ticket_dollars': 'tickets',
                                'Continent': 'continent'}, inplace=True)

selected_cities['norm_rank'] = (selected_cities['rank'] - selected_cities['rank'].min()) / (selected_cities['rank'].max() - selected_cities['rank'].min())

######################################################Data##############################################################

indicator_names = ['rank', 'arrivals', 'growth', 'income']

summable_indicators = ['arrivals', 'income', 'hotel_price', 'tickets']

######################################################Interactive Components############################################

city_options = [dict(label=city, value=city) for city in selected_cities.index]

indicator_options = [dict(label=indicator, value=indicator) for indicator in indicator_names]

##################################################APP###############################################################

app = dash.Dash(__name__)
server = app.server#!!!!!!!!!!!!!!
app.layout = html.Div([

    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('logo3.jpg'),style={'height':'50%'}),
        ], className='column5'),

        html.Div([
            html.H1('World Tour Simulator'),
        ], className='column6'),

        html.Div([
            html.Img(src=app.get_asset_url('Rotating_globe.gif'), style={'height':'50%', 'align': 'right'}),
        ], className='column7'),

    ], className='Title row2'),

    html.Div([

        html.Div([
            html.Label('Select Cities to Visit'),
            dcc.Dropdown(
                id='city_drop',
                options=city_options,
                value=list(np.random.choice(selected_cities.index, 10, replace=False)),
                multi=True,
            ),

            html.Br(),

            html.Label('Select Indicators'),

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
        #]),
    #]),
            html.Button('Submit', id="button")

        ], className='column1 pretty'),

    html.Div([

            html.Div([html.Label('Cities above the minimum rank:'), html.Label(id='indic_0')], className='mini pretty'),        
          
            html.Div([

                html.Div([html.Label(id='indic_1')], className='mini pretty'),
                html.Div([html.Label(id='indic_2')], className='mini pretty'),
                html.Div([html.Label(id='indic_3')], className='mini pretty'),
                #html.Div([html.Label(id='indic_4')], className='mini pretty'),
                #html.Div([html.Label(id='indic_5')], className='mini pretty'),
            ])

        ], className='column2')

    ], className='row'),

    html.Div([dcc.Graph(id='scattergeo')], className='column3 pretty'),

    html.Div([

        html.Div([dcc.Graph(id='bar_graph')], className='bar_plot pretty'),
        html.Div([dcc.Graph(id='bubbles')], className='bar_plot pretty')

    ])

])

######################################################Callbacks#########################################################

@app.callback(
    [
        Output("bar_graph", "figure"),
        Output("scattergeo", "figure"),
        Output("bubbles", "figure")
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

    data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=indicator,marker=dict(color='red')))

    layout_bar = dict(title=dict(text=indicator.title() + ' per City',font=dict(color='#ffc40e')),
                  yaxis=dict(title=indicator.title() + ' Value',color='#ffc40e'),
                  xaxis=dict(title='Cities', color='#ffc40e'),
                  paper_bgcolor='#000000',
                  plot_bgcolor='#000000',
                  )

    #############################################Second ScatterGeo######################################################

    data = [[distance(i, j) for j in new_selection.index] for i in new_selection.index]

    run = ga_search(data)
    print(run)

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

    df = df.merge(selected_cities['rank'], left_on='name_city', right_on='city', how='left')

    map_data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == 'Generation_0',"x_coordinate"] , 
                     lon=df.loc[df.loc[:,"generation"] == 'Generation_0',"y_coordinate"] ,
                     hoverinfo = 'text',
                     text = df.loc[df.loc[:,"generation"] == 'Generation_0',"name_city"],
                     mode="lines+markers",
                     line=dict(
                         width=1,
                         color="blue",
                     ),
                     marker=dict(
                         size=12,
                         #color="red",
                         colorscale='Reds',
                         cmin = df['rank'].min(),
                         color=df.loc[df.loc[:, "generation"] == 'Generation_0', "rank"],
                         cmax = df['rank'].max(),
                         reversescale = True,
                         colorbar=dict(title="Tourism Ranking<br>2018")
                            ))]
    
    map_layout=go.Layout(
        title=dict(text='Optimized World Tour',
                                   xanchor='center',
                                   y=0.95,
                                   x=0.5,
                                   yanchor='top'),
        #title_text="Optimized World Tour",
        hovermode="closest",
        #width=1400,
        margin=go.layout.Margin(
            l=5,
            r=5,
            b=10,
            t=50,
            #pad=4
        ),
        font=dict(color='#ffc40e', size=14, family='Open Sans, sans-serif'),
        paper_bgcolor = 'black',
        geo = dict(
            showland = True,
            showcountries = True,
            showocean = True,
            countrywidth = 0.5,
            landcolor = 'rgb(204, 204, 204)',
            countrycolor = 'rgb(204, 204, 204)',
            lakecolor = 'rgb(0, 255, 255)',
            bgcolor = 'black',
            oceancolor = 'black'
            ),
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])])
    
    map_frames=[go.Frame(
        data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == k,"x_coordinate"] , 
                     lon=df.loc[df.loc[:,"generation"] == k,"y_coordinate"] ,
                     text = df.loc[df.loc[:,"generation"] == k,"name_city"],
                     mode="lines+markers",
                     line=dict(width=((df.loc[df.loc[:,"generation"] == k,"norm_distance"].iloc[0])+0.1)*8, color="blue"),
                     marker=dict(
                         size=12,
                         #color="red",
                         colorscale='Reds',
                         cmin=df['rank'].min(),
                         color=df.loc[df.loc[:, "generation"] == k, "rank"],
                         cmax=df['rank'].max(),
                         reversescale=True,
                         colorbar=dict(title="Tourism Ranking<br>2018")))])

        for k in df.loc[:,"generation"].unique()]

############################################Bubble Scatter Plot##########################################################
    bubble_x = list(new_selection['hotel_price'])
    bubble_y = list(new_selection['tickets'])
    bubble_size = list(new_selection['rank'].pow(-1))
    bubble_color= list(new_selection['continent'])
    
    bubble_data = []

    for i in range(len(bubble_x)):
        bubble_data.append(dict(x=bubble_x[i],
                                y=bubble_y[i],
                                color=bubble_color[i],
                                mode='markers',
                                marker={
                                        'size':bubble_size[i],
                                        'sizemode':'area',
                                        'sizeref':2.*max(bubble_size)/(40.**2),
                                        'sizemin':4,
                                }))

    #bubble_data = [{'x':bubble_x,
    #                'y':bubble_y,
    #                #'color': bubble_color,
    #                'mode':'markers',
    #                'marker':{
    #                    'size':bubble_size,
    #                    'sizemode':'area',
    #                    'sizeref':2.*max(bubble_size)/(40.**2),
    #                    'sizemin':4,
    #                    'color': bubble_color,
    #                    },     
    #              }]

    bubble_layout = dict(
                        title=dict(text='Lodging and Transportation Costs (US$)',
                                   xanchor='center',
                                   y=0.9,
                                   x=0.5,
                                   yanchor='top'),
                        xaxis=dict(
                            title='Hotel Room Price',
                            gridcolor='white',
                            gridwidth=2,
                        ),
                        yaxis=dict(
                            title='Public Transportation Single Round Fare',
                            gridcolor='white',
                            gridwidth=2,
                        ),
                        paper_bgcolor='#000000',
                        plot_bgcolor='rgb(243, 243, 243)',
                        font=dict(color='#ffc40e', size=14, family='Open Sans, sans-serif')
                        )
    


#########################################################################################################################

    return go.Figure(data=data_bar, layout=layout_bar), \
           go.Figure(data=map_data, layout=map_layout, frames=map_frames), \
           go.Figure(data=bubble_data, layout=bubble_layout)

@app.callback(
    [
        Output("indic_0", "children"),
        Output("indic_1", "children"),
        Output("indic_2", "children"),
        Output("indic_3", "children"),
        #Output("indic_4", "children"),
        #Output("indic_5", "children"),
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
    cities_max = cities.max()
    cities_min = cities.min()

    value_1 = round(cities_sum[summable_indicators[0]]/1000000,0)
    value_2 = round(cities_sum[summable_indicators[1]],2)
    value_3 = round(cities_avg[summable_indicators[2]],2)
    value_4 = round(cities_max[summable_indicators[2]], 2)
    value_5 = round(cities_min[summable_indicators[2]], 2)
    
    cities_selection = [str(x) for x in cities.index]
    cities_selection.sort()
    cities_selection = ', '.join(cities_selection)

    return cities_selection,\
           ' Average Hotel Price: $' + str(value_3), \
           ' Maximum Hotel Price: $' + str(value_4), \
           ' Minimum Hotel Price: $' + str(value_5),

           #str(summable_indicators[0]).title() + ' sum: ' + str(value_1) + 'million people',\
           #str(summable_indicators[1]).title() + ' sum: $' + str(value_2) + ' billion',\
           #str(summable_indicators[2]).title() + ' Average Hotel Price: $' + str(value_3), \
           #str(summable_indicators[2]).title() + ' Maximum Hotel Price: $' + str(value_4), \
           #str(summable_indicators[2]).title() + ' Minimum Hotel Price: $' + str(value_5),



if __name__ == '__main__':
    app.run_server(debug=True)