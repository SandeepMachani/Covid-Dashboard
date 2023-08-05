from dash import dcc, html, Input, Output, no_update, callback
import dash
import plotly.express as px
from . import Pre_processing
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


# app = dash.Dash('TEST')  # the name of the folder containing your code and static folder.
# app.css.append_css({'external_url': '/static/ScatterC.css'})
# app.server.static_folder = 'static'
dash.register_page(__name__,
    relative_path='/Scatter',
    title='Scatter',
    name='Scatter Plot')

dfWorld = Pre_processing.dfWorld
# Choose all the data that are normalized
ColumnNames = [     'Total Cases Per Million', "Total Deaths Per Million", 'New Cases Per Million' , 'New Deaths Per Million' , 
                    'Total Vaccinations Per Hundred', 'Total tests per thousand', 'Stringency Index']
                    # 'gdp_per_capita' , 'Population', 'Stringency Index']

#______________________________________________________________________________________________________________________________________

# dash.register_page(__name__)
# app = Dash(__name__) 

layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Markdown("Use the filters such as Continents, Stringency, and Population and select the desired country on the bubble plot to see Time-Series data on the right.",
                style = {'font-size': '100%', 'background-color':'#d9ead3', "text-align": "center"}),
            
            dcc.Markdown("To have animation-like movement in the bubble plot click on the slider and use singular arrow keys of left or right.",
                style = {'font-size': '100%', 'background-color':'#d9ead3', "text-align": "center"})
        ], width = {"size" : 10, "offset": 1}, style = {'font-size': 23}),
        ]),
    dbc.Row([

        dbc.Col([
            dbc.Row([
                dbc.Col(html.Label("Choose the Parameter of X Axis: "), width= {"size": 4, "offset":1}),
                dbc.Col(
                    dcc.Dropdown(
                        ColumnNames,
                        'Total Cases Per Million',
                        id='crossfilter-xaxis-column',
                        style = {"background-color": '#A5D6A7', "text-align" : "center",} 
                    ), width = {"size": 6},)
                ]),
            dbc.Row([
                dbc.Col(html.Label("Choose the Scale of X Axis: "), width= {"size": 4, "offset":1}),
                dbc.Col(
                    dcc.RadioItems(
                        ['Linear', 'Log'],
                        'Linear',
                        id='crossfilter-xaxis-type', #inputStyle= {"margin-right": "10px", "margin-left": "70px"},
                        labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                        'padding':'0.5rem 0.80rem',
                        'border-radius':'2rem'},
                        style={'display':'flex', 'font-size': '85%'}, 
                        inputStyle={'cursor':'pointer'},                         
                    ), width = {"size": 6},
                )
            ]),
            dbc.Row([
                dbc.Col(html.Label("Choose the Population Filter: "), width= {"size": 4, "offset": 1}),
                dbc.Col(
                    dcc.Dropdown(
                        ["More than 100M", "More than 50M", "More than 10M", "No Filter"],
                        value = "No Filter",
                        id = "filter-Population",
                        style = {"background-color": '#A5D6A7', "text-align" : "center",}
                    ), width = {"size": 6})
            ]),
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        dbc.Col([
            
            dbc.Row([
                dbc.Col(html.Label("Choose the Parameter of Y Axis: "), width= {"size": 4}),
                dbc.Col(
                    dcc.Dropdown(
                    ColumnNames,
                    'Total Deaths Per Million',
                    id='crossfilter-yaxis-column',
                    style = {"background-color": '#A5D6A7', "text-align" : "center",})
                    , width= {"size": 6} )

            ]),
            
            dbc.Row([
                dbc.Col(html.Label("Choose the Scale of Y Axis: "), width= {"size": 4}),
                dbc.Col(
                    dcc.RadioItems(
                    ['Linear', 'Log'],
                    'Linear',
                    id='crossfilter-yaxis-type',# inputStyle= {"margin-right": "10px", "margin-left": "70px"},
                    # labelStyle={'display': 'inline-block', 'marginTop': '5px', },),
                        labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                        'padding':'0.5rem 0.80rem',
                        'border-radius':'2rem'},
                        style={'display':'flex', 'font-size': '85%'}, 
                        inputStyle={'cursor':'pointer'},),
                    
                    width= {"size": 6} ),
            ]),
            
            dbc.Row([
                dbc.Col(html.Label("Choose the Stringency Filter: "), width= {"size": 4}),
                dbc.Col(
                    dcc.Dropdown(
                    ['Stringency >25', 'Stringency >50', 'Stringency >75', "No Filter"],
                    'No Filter',
                    id ='filter-stringency',
                    style ={'color': 'black' , "background-color": '#A5D6A7' , "text-align" : "center",}
                ), width= {"size": 6}), 
            ]),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),   
    ], style={
        'padding': '10px 5px',
        "background-color": "#d9ead3"
    }),

    dbc.Col([
        dbc.Row(html.Br()),
        
        dbc.Row(
            html.Div("Bubble Plot for Different Countries in a Certain Week Starting From 2020"),
            style={'font-weight': 'bold', "text-align": "center", 'font-size': 20}
        ),
        
        dbc.Row(               
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                clickData={'points': [{'customdata': 'Norway' , 'hovertext': 'Norway'}]},)
        ),
        
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20' , "background-color": "#d9ead3"}),
    
    dbc.Col([
        dbc.Row(html.Br()),
        
        dbc.Row(
            html.Div("Time-Series Plots for Selected Country For the X and Y Axis Respectively"),
            style={'font-weight': 'bold', "text-align": "center", 'font-size': 20}
        ),
        
        dbc.Row(
            dcc.Graph(id='x-time-series'),
        ),
        
        dbc.Row(        
            dcc.Graph(id='y-time-series'),
        ),
        dbc.Tooltip("Filter population of the shown countries in the scatter plot below"
                ,target = "filter-Population",
                autohide=False, style = {"font-size":15}, placement = "top"),
        
        dbc.Tooltip("Filter the data shown below with certain Stringency Index values. Red dots are the points where stringency is more than specified"
                ,target = "filter-stringency",
                autohide=False, style = {"font-size":15}, placement = "top"),
        
    ], style={'display': 'inline-block', 'width': '49%', "background-color": "#d9ead3"}),
    
#Slider
    html.Div(dcc.Slider(
        min=dfWorld['New Time'].min(),
        max=dfWorld['New Time'].max(),
        step= 1,
        id='crossfilter-year--slider',
        marks={i: '{}'.format('Week: ' + str(i)) for i in range(dfWorld['New Time'].min(),dfWorld['New Time'].max(),20)},
        value= 20,
        tooltip={"placement": "bottom", "always_visible": True},
        
    ), style={ 'width': '49%', 'padding': '0px 20px 20px 20px', "background-color": "#d9ead3"}),
    
    html.Div(style={"background-color": "#d9ead3"})
], style = {"height": "100vh" , "font-size" : 20, "background-color": "#d9ead3"})

#function for filtering Population:
def Population_filter(dff, filter):
    df = dff
    if filter == "More than 100M":
        df = dff[dff['Population'] > 100000000]
        
    if filter == "More than 50M":
        df = dff[dff['Population'] > 50000000]
        
    if filter == "More than 10M":
        df = dff[dff['Population'] > 10000000]
        
    if filter == "No Filter":
        df = dff
        
    return df

@callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-year--slider', 'value'),
    Input('filter-Population', 'value'),
    Input('filter-stringency', 'value')
    )

def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type, year_value,
                 filter_Population, filter_stringency):
    
    dfScatter = dfWorld[dfWorld['New Time'] == year_value]
    
    dfScatter = stringency_filter(dfScatter, filter_stringency)
    
    dfScatter = Population_filter(dfScatter, filter_Population)
    
    fig = px.scatter(dfScatter, x =  xaxis_column_name, y = yaxis_column_name,
            size = "Population" ,color="Continent", size_max= 55,
            # template = "seaborn",
            hover_name = "Location", #hover_data=["continent", "pop"]
            # custom_data=["Location"]
            )

    fig.update_layout(hoverlabel=dict(bgcolor= "#d9ead3" , font_size=16, font_family="Times"))

    # fig.update_traces(customdata = find_country(dfScatter))
    
    fig.update_layout(paper_bgcolor= "#d9ead3")
    
    fig.update_layout(font = {"size": 15}, yaxis = dict(tickfont = dict(size=15)), xaxis = dict(tickfont = dict(size=15)),)

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    # fig.update_layout(clickmode='event+select')

    return fig

#function to create figures on the right ( time series ):
def create_time_series(dff, axis_type, title, parameter, filter_stringency):
    
    # fig = make_subplots(y_title= axis_type, x_title= "Date")
    fig = go.Figure()
    
    # fig = px.scatter(dff, x='Date', y= parameter,  template = "seaborn",)
    fig.add_trace(go.Scatter(x=dff['Date'], y= dff[parameter],  mode = "lines+markers", name = "Unfiltered Data" ,marker = dict(color = "blue")))
    
    stringencydff = stringency_filter(dff, filter_stringency)
    
    if(filter_stringency!= "No Filter"):
        fig.add_trace(go.Scatter(x=stringencydff['Date'], y= stringencydff[parameter], mode = "markers", name = "Stringency Filtered" ,marker = dict(color = "red")))
    
    # fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_layout(paper_bgcolor= "#d9ead3")
    
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.70, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title, )
    
    fig.update_layout(font = {"size": 13}, yaxis = dict(tickfont = dict(size=15)), xaxis = dict(tickfont = dict(size=15)),)
    
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig

#function for stringency filter:
def stringency_filter(dff, filter):
    
    dfmask = dff
    if filter == 'No Filter':
        return dfmask

        
    elif filter == 'Stringency >25':
        mask = (dff["Stringency Index"] > 25)
        
    elif filter == 'Stringency >50':
        mask = (dff["Stringency Index"] > 50)
        
    elif filter == 'Stringency >75':
        mask = (dff["Stringency Index"] > 75)
        
    # elif filter == 'Stringency > 75':
    #     mask = (dff["Stringency Index"] > 75)       
        
    dfmask = dff[mask]
    return dfmask

@callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', "clickData"),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('filter-stringency', 'value'),)

def update_y_timeseries(clickData, xaxis_column_name, axis_type, filter_stringency):
    country_name = clickData['points'][0]['hovertext']
    dff = dfWorld[dfWorld['Location'] == country_name]
    # dff = dff[xaxis_column_name]
    title = '<b>{}</b><br>{}<br>{}'.format(country_name, xaxis_column_name, filter_stringency)
    return create_time_series(dff, axis_type, title, xaxis_column_name, filter_stringency)


@callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', "clickData"),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('filter-stringency', 'value'),)
def update_x_timeseries(clickData, yaxis_column_name, axis_type, filter_stringency):
    dff = dfWorld[dfWorld['Location'] == clickData['points'][0]['hovertext']]
    # dff = dff[yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name, yaxis_column_name, filter_stringency)



# if __name__ == '__main__':
# 	app.run_server(debug=True)

