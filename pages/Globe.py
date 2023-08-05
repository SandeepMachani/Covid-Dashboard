# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import dcc, Output, Input, html, callback  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import plotly.express as px
import pandas as pd                        # pip install pandas
import dash
from . import Pre_processing
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# incorporate data into app
# Source - https://www.cdc.gov/nchs/pressroom/stats_of_the_states.html
df = Pre_processing.df


dfWorld = Pre_processing.dfWorld[Pre_processing.dfWorld['Date']> "2020-03-10"]
dfAsia = Pre_processing.dfAsia[Pre_processing.dfAsia['Date']> "2020-03-10"]
dfEurope = Pre_processing.dfEurope[Pre_processing.dfEurope['Date']> "2020-03-10"]
dfAfrica = Pre_processing.dfAfrica[Pre_processing.dfAfrica['Date']> "2020-03-10"]
dfNA =  Pre_processing.dfNA[Pre_processing.dfNA['Date']> "2020-03-10"]
dfSA =   Pre_processing.dfSA[Pre_processing.dfSA['Date']> "2020-03-10"]
dfOceania =Pre_processing.dfOceania[Pre_processing.dfOceania['Date']> "2020-03-10"]


# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__,
    relative_path='/Globe',
    title='Globe',
    name='Global Data')

globe = dcc.Graph(id = "globe", figure={}, clickData={'points': [{'customdata': 'Norway' , 'hovertext': "Norway"}]},)

parameter_choice = ["Total Cases", "New Cases", "Total Deaths", "New Deaths", "New Cases Per Million",
                    "New Deaths Per Million" , "Total Vaccinations",  "Stringency Index",]


parameter = dcc.Dropdown(options= parameter_choice,
                        value='New Cases',  # initial value displayed when page first loads
                        clearable=False, style = {"background-color": '#A5D6A7', "text-align" : "center"})
continents = dcc.Dropdown(options = ['Asia' , 'Europe' , 'Africa' , 'North America', 'South America', 'World', 'World 2D'],
                        value = 'World', style = { "background-color": '#A5D6A7', "text-align" : "center"})
graph = dcc.Graph(figure={})

country_options = dfWorld["Location"].unique()
country_options.sort()

last_countries = []
filter_happened = True
filter_continent = False
region_filter_happened = True
statistic_filter_happened = False

layout = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Col(html.Div([globe]), style = {'background-color': "#d9ead3"}),  
        ], width = {"size": 6} , style = {'background-color': "#d9ead3"} ),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.P("Use the filters and clicking on the map to choose countries. You can also zoom in and change to different styles of map using continent filter. Pay in mind, adding countries by clicking the map is not available when you are on the statistic filter.",
                       style = {'font-size': '18', 'background-color':'#d9ead3'})
                ], width = {'size':11})
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div("Countries: ")
                ], width = {"size":  2} ),
                
                dbc.Col([
                    dcc.Dropdown(id = "countries", options = country_options,
                                 value  = ["Norway", "Sweden", "Denmark", "Finland", "Iceland"], multi=True,
                                 style = {"background-color": '#A5D6A7', "font-color": 'black',  "text-align" : "center"})
                ], width = {"size": 9}),
                
            ], justify = "left"),
            
            dbc.Row([
                    html.Br(),
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div("Continent: "), width = 2
                ),
                dbc.Col(
                    html.Div([continents]), width = 3
                ),
                dbc.Col(
                    html.Div("Parameter: "), width = 2
                ),
                dbc.Col(
                    html.Div([parameter]), width = 4
                ),
            ], justify = "left"),
            

            
            # dbc.Row( html.Div("Use the different filters including Continents, Regions, Statistic Filters, and clicking on the map below to filter the countries shown on the graph")
            #     ),

        
            dbc.Row([
                dbc.Col(html.Div("Filtering by: "), width = {"size" : 2}),
                dbc.Col([

                    dcc.RadioItems(id = "choose_filter", options = ["Regions", "Statistic Filters"],
                    value = "Regions",
                    # style = {"font-color": 'black'}, inputStyle= {"margin-right": "10px", "margin-left": "60px"})
                        labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                        'padding':'0.2rem 0.80rem',
                        'border-radius':'2rem'},
                        style={'display':'flex', 'font-size': '85%'}, 
                        inputStyle={'cursor':'pointer'}, )
                    
                ], width = 8)
            ]),
                    
        dbc.Row( id = 'filter_chosen'
            ),
        dbc.Row(
            html.Div("Time-Series Graph for Chosen Countries:",
                     style={'font-weight': 'bold', "text-align": "center", 'font-size': 25})
        ),
        dbc.Row(
            dbc.Col(dcc.Graph("scatter"), style = {"background-color": '#d9ead3', 'height': 600})
        ),
        
            ], width = {"size" : 6  , 'background-color': "#d9ead3"}, style = {'font-family': 'sans-serif', "font-size": 20}
       
        ),
    
    html.P(id='placeholder') ,
    
    ]),
],  style={"height": "100vh", 'background-color': "#d9ead3"}, )



#callback to adjust the layout based on chosen filters:
@callback(
    Output("filter_chosen", 'children'),
    Input('choose_filter', 'value'),
)
def choose_filter(chosen_filter):
    global region_filter_happened
    global statistic_filter_happened
    
    if chosen_filter == "Regions":
        region_filter_happened = True
        statistic_filter_happened = False

        row_content = [
                dbc.Col(html.Div("Regions: "), width = {"size" : 2}),
                dbc.Col([

                    dcc.Dropdown(id = "region_filter", options = ["World","Nordic Countries", "Middle East", "South East Asia",
                                                              "South Asia", "Western Europe", "Southern Europe",
                                                              "North America", "South America", "Africa"],
                    value = "World",
                    style = {"background-color": '#A5D6A7', "font-color": 'black', "text-align" : "center"})
                ], width = 9),
                
                html.P(id='statistic_filter') ,
                html.P(id = 'parameter_measure')
            ]
        
        return row_content
    
    elif chosen_filter == "Statistic Filters":
        statistic_filter_happened = True
        region_filter_happened = False
        
        row_content = [
                dbc.Col(html.Div("Filters: "), width = {"size" : 2}),
                dbc.Col([

                    dcc.Dropdown(id = 'statistic_filter', options = ["Human Development Index", "Diabetes Prevalence"
                                                                     ,"Handwashing Facilities", "Gdp Per Capita"],
                    value = "Gdp Per Capita",
                    style = {"background-color": '#A5D6A7', "font-color": 'black'})
                ], width = 5),
                dbc.Col(html.Div("Scale: "), width = {"size" : 1}),
                dbc.Col([
                    dcc.Dropdown(id = "parameter_measure", options = ["None", "High", "Moderate", "Low"],
                    value = "None",
                    style = {"background-color": '#A5D6A7', "font-color": 'black'})
                ], width = {"size": 2}),
                html.P(id='region_filter') 
 
        ]
        return row_content

#callback to check if region changed or not:
@callback(
    Output('placeholder', 'children'),
    Input('region_filter', 'value'),
)
def counter_for_region_filter(region_filter):
    global filter_happened
    filter_happened = True
    return (None)

#callback to change globe graph based on continents and achieve click data:
@callback(
    Output("globe", 'figure'),
    Input(parameter, 'value'),
    Input(continents, 'value'),
    # Input("countries", 'value')
)
def update_graph(column_name, continents): 
    
    if(continents == "World"):
        dataframe = dfWorld
        scope = "world"
        projection_type = "orthographic"
    
    elif(continents == "World 2D"):
        dataframe = dfWorld
        scope = "world"
        projection_type = None
        
    elif(continents == "Europe"):
        dataframe = dfEurope
        projection_type = None
        scope = "europe"

    elif(continents == "Asia"):
        dataframe = dfAsia
        projection_type = None
        scope = "asia"
        
    elif(continents == "Africa"):
        dataframe = dfAfrica
        projection_type = None
        scope = "africa"     
        
    elif(continents == "North America"):
        dataframe = dfNA
        projection_type = None
        scope = "north america"    

    elif(continents == "South America"):
        dataframe = dfSA
        projection_type = None
        scope = "south america"        
        
    fig = px.choropleth(data_frame = dataframe,
                            locations= 'Iso Code',
                            height= 810,
                            # width = 550,
                            color= column_name,
                            animation_frame ='Date Weekly',
                            scope = scope,
                            color_continuous_scale='Blues',
                            template = "plotly",  
                            hover_name = "Location",
        
                            )
    
    fig.update_geos( bgcolor = "#d9ead3", projection_type = projection_type, lakecolor  = "#d9ead3")
    fig.update_layout(paper_bgcolor= "#d9ead3")
    fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 20)))
    fig.update_layout(title_font_size = 25, uniformtext_minsize=25, uniformtext_mode    ='hide', )

    fig.update_layout(
    margin={'r':0,'t':30,'l':0,'b':0},
    coloraxis_colorbar={})
    
    return fig


#Fucntion to check region filters:
def check_region(region_filter, countries_input):
    global filter_happened
    
    if region_filter == "World":
        countries = countries_input
        filter_happened = False
        return countries
    
    elif region_filter == "Nordic Countries":
        countries = ["Norway", "Sweden", "Denmark", "Finland", "Iceland"]
        filter_happened = False
        return countries
    
    elif region_filter == "Middle East":
        countries = ["Iran", "Iraq", "Saudi Arabia", "Turkey", "Egypt"]
        filter_happened = False
        return countries
    
    elif region_filter == "South East Asia":
        countries = ["Indonesia", "Philippines", "Malaysia", "Thailand", "Singapore", "Myanmar", "Vietnam"]
        filter_happened = False
        return countries
    
    elif region_filter == "South Asia":
        countries = ["India", "Pakistan", "Afghanistan", "Bangladesh"]
        filter_happened = False
        return countries
    
    elif region_filter == "Western Europe":
        countries = ["Germany", "United Kingdom", "Belgium", "Netherlands"]
        filter_happened = False
        return countries

    elif region_filter == "Southern Europe":
        countries = ["Italy", "Portugal", "Spain"]
        filter_happened = False
        return countries 
    
    elif region_filter == "North America":
        countries = ["United States", "Canada", "Mexico"]
        filter_happened = False
        return countries  
     
    elif region_filter == "South America":
        countries = ["Brazil", "Argentina", "Chile", "Colombia", "Peru"]
        filter_happened = False
        return countries     

    elif region_filter == "Africa":
        countries = ["Egypt", "Algeria", "Morocco", "South Africa"]
        filter_happened = False
        return countries   
    
#Function for statistic filters:
def check_statistic(statistic_filter, measure):
    global filter_happened
    filter_happened = True
    
    if statistic_filter == "None":
        countries_options = dfWorld['Location'].unique()
        filter_happened = False
        return countries_options

    elif statistic_filter != "None":
        # countries = countries_input
        filter_happened = False
        countries_options = measure_calculation(statistic_filter, measure)
        return countries_options

#Function for measures of statistic filters:
def measure_calculation(statistic_filter, measure):
    scope = dfWorld[statistic_filter].max() - dfWorld[statistic_filter].min()
    scope_med = None
    scope_low = None 
    
    if statistic_filter == "Gdp Per Capita":
        scope_low = dfWorld[statistic_filter].max()/4 
        scope_med = dfWorld[statistic_filter].max()*2/6       
        
    if statistic_filter == "Human Development Index":
        scope_low = dfWorld[statistic_filter].max()/2 
        scope_med = dfWorld[statistic_filter].max()*4/5
        
    elif statistic_filter == "Handwashing Facilities":
        scope_low = dfWorld[statistic_filter].max()/3
        scope_med = dfWorld[statistic_filter].max()*3/5
        
    elif statistic_filter == "Diabetes Prevalence":
        scope_low = dfWorld[statistic_filter].max()/3 
        scope_med = dfWorld[statistic_filter].max()*3/5       
        
    if measure == "None":
        countries = dfWorld['Location'].unique()
        return countries
    
    if measure == "Low":
        countries = dfWorld[dfWorld[statistic_filter] < scope_low]['Location'].unique()

        return countries
    
    if measure == "Moderate":
        mask = (dfWorld[statistic_filter] < scope_med) & (dfWorld[statistic_filter] > scope_low)
        countries = dfWorld[mask]['Location'].unique()

        return countries
    
    if measure == "High":
        countries = dfWorld[dfWorld[statistic_filter] > scope_med]['Location'].unique()

        return countries
        
#callback for the dropdown countries chosen from the globe graph and filters
@callback(
    Output('countries', 'value'),
    Input("globe", "clickData"),
    Input('countries', 'value'),
    Input('region_filter', 'value'),
    Input('countries', 'options'),
    Input('statistic_filter','value'),
    Input("parameter_measure", 'value')
    )

def input_dropdown(clickData, countries, region_filter, country_options, statistic_filter, measure):
    country_name = clickData['points'][0]['hovertext']
    
    global last_countries
    global filter_happened
    global filter_continent 
    global statistic_filter_happened
    global region_filter_happened
    
    countries_options = []
    
    #Filter for continents
    if filter_continent == True:
        for c in countries:
            if c in country_options:
                countries_options.append(c)
                
        filter_continent = False
        return(countries_options) 
    
    #Filter for region choices if it happened: 
    if region_filter_happened == True:
        if filter_happened == True:
            return (check_region(region_filter, countries))  
        
    countries_statistic = []

    #Filter for statistic filter if it happened:
    if statistic_filter_happened == True:
        for c in countries:
            if c in check_statistic(statistic_filter, measure):      
                countries_statistic.append(c)
        return (countries_statistic)
    
    #Add country from add click:
    if country_name not in countries:
        countries.append(country_name)
    
    #remove last country to avoid getting stuck with the clicked country
    if countries == last_countries:
        countries.pop()
        return countries
    
    last_countries = countries
    
    return countries


#callback to make options of dropdown interactive with filters chosen:
@callback(
    Output("countries", 'options'),
    Input(continents, 'value'),
    Input('statistic_filter','value'),
    Input("parameter_measure", 'value')
)
def update_dropdown(continents, statistic_filter, measure):
    
    global filter_continent
    global statistic_filter_happened
    
    # make options interactive and change with chosen continents:
    if(continents == "World"):
        dataframe = dfWorld
        
    elif(continents == "World 2D"):
        dataframe = dfWorld
              
    elif(continents == "Europe"):
        dataframe = dfEurope
        filter_continent = True
        
    elif(continents == "Asia"):
        dataframe = dfAsia
        filter_continent = True
       
    elif(continents == "Africa"):
        dataframe = dfAfrica
        filter_continent = True
        
    elif(continents == "North America"):
        dataframe = dfNA
        filter_continent = True

    elif(continents == "South America"):
        dataframe = dfSA
        filter_continent = True

    options = dataframe["Location"].unique()
    
    options_valid_statistic = []
    options.sort()
    
    if statistic_filter_happened == True:
        for c in options:
            if c in measure_calculation(statistic_filter, measure):
                options_valid_statistic.append(c)
        return(options_valid_statistic)

    return options


#callback to make a line graph based on filters
@callback(
    Output("scatter", 'figure'),
    #Input(mytitle, 'children'),
    Input(parameter, 'value'),
    Input("countries", 'value'),
)
def update_scatter(column_name, countries, ): 
    
    countries = countries

    fig = make_subplots(y_title= column_name, x_title=" Date (Weeks from the start of 2020) ")
    for country in countries:
        fig.add_trace(go.Scatter(x = dfWorld[dfWorld['Location'] == country]["Date"] ,
                                 y = dfWorld[dfWorld['Location'] == country][column_name],
                        mode='lines', name = str(country)))
        
    fig.update_layout(paper_bgcolor= "#d9ead3")
    fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
    fig.update_layout(    font=dict(size=15), )

    return fig
    
# if __name__=='__main__':
#     app.run_server(debug=True, port=8054)
