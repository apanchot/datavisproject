#!/usr/bin/env python
# coding: utf-8

# # TSP Problem
#
# ## 1. Defining a function to compute de distances between two gps coordinates.

# In[2]:


from math import sin, cos, sqrt, atan2, radians
print("hello")
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


# # 2. Importing a dataframe that contains latitude and longitude coordinates of 15,493 cities from around the world.

# In[3]:


import pandas as pd
cities_coordinates = pd.read_csv('worldcities.csv')


# # 3. Selecting a group of 22 big cities

# In[4]:


selected = ['Tokyo','New York','Mexico City','Rio de Janeiro','Los Angeles','Buenos Aires','Rome','Lisbon','Paris',
            'Munich','Changping','Delhi','Sydney','Moscow','Istanbul','Cape Town','Madrid','Seoul','London','Bangkok',
            'Toronto','Dubai']
selected_cities = cities_coordinates.loc[cities_coordinates['city'].isin(selected),['city','country','lat','lng']]
selected_cities.reset_index(inplace = True, drop = True)
selected_cities = selected_cities.loc[0:21,:]
selected_cities = selected_cities.drop('country', axis = 1)
selected_cities.set_index('city', inplace = True)


# In[5]:


selected_cities.head()


# # 4. Computing the distances between each of them.

# In[6]:


distances = [[distance(i,j) for j in selected_cities.index] for i in selected_cities.index]


# In[8]:


distances


# # 5. After running the GA Algorithm, a particular run has been selected to be plotted.

# In[9]:


run = pd.read_excel('run_13.xlsx')


# # 6. Preparing the dataframe

# In[6]:


def path(x):
    best_fitness_aux = run.loc[x,'Fittest'].replace(',','').replace('[','').replace(']','').split(' ')
    path_best_fitness = [int(i) for i in best_fitness_aux]
    path_best_fitness = path_best_fitness + [path_best_fitness[0]]
    return path_best_fitness


# In[7]:


generation = lambda x: ['Generation_'+str(run.loc[x,'Generation'])]*len(path(x))


# In[8]:


all_path = []
all_generation = []
for i in run.loc[:,'Generation']:
    all_path = all_path + path(i)
    all_generation = all_generation + generation(i)


# In[9]:


all_generation = pd.Series(all_generation)
all_path = pd.Series(all_path)


# In[10]:


x_coordinate = [selected_cities.iloc[i,0] for i in all_path]
y_coordinate = [selected_cities.iloc[i,1] for i in all_path]
name_city = [selected_cities.index[i] for i in all_path]
x_coordinate = pd.Series(x_coordinate)
y_coordinate = pd.Series(y_coordinate)
name_city = pd.Series(name_city)


# In[11]:


df = pd.concat([all_generation, all_path, name_city, x_coordinate, y_coordinate], axis = 1)
df.columns = ['generation', 'city', 'name_city', 'x_coordinate','y_coordinate']


# # 7. Plotting

# In[13]:


import plotly.graph_objects as go

# Create figure
fig = go.Figure(
    data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == 'Generation_0',"x_coordinate"] ,
                     lon=df.loc[df.loc[:,"generation"] == 'Generation_0',"y_coordinate"] ,
                     hoverinfo = 'text',
                     text = df.loc[df.loc[:,"generation"] == 'Generation_0',"name_city"],
                     mode="lines+markers",
                     line=dict(width=1, color="blue"),
                     marker=dict(size=4, color="red"))],
    layout=go.Layout(
#         xaxis=dict(range=[-60,60], autorange=False, zeroline=False),
#         yaxis=dict(range=[-150,160], autorange=False, zeroline=False),
        title_text="TSP Problem", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None]),
                                   dict(label="Pause",
                                        method="animate",
                                        args=[None])])]),
    frames=[go.Frame(
        data=[go.Scattergeo(lat=df.loc[df.loc[:,"generation"] == k,"x_coordinate"] ,
                     lon=df.loc[df.loc[:,"generation"] == k,"y_coordinate"] ,
                     text = df.loc[df.loc[:,"generation"] == k,"name_city"],
                     mode="lines+markers",
                     line=dict(width=1, color="blue"),
                     marker=dict(size=4, color="red"))])

        for k in df.loc[:,"generation"].unique()]
)

fig.show()


# In[ ]:
