from dash import dcc, html, Input, Output, no_update, dash_table, callback
import dash
import plotly.express as px
import pandas as pd
import numpy as np
from . import Pre_processing
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import itertools
import dash_bootstrap_components as dbc
# from dash_table.Format import Format, Scheme
from plotly.subplots import make_subplots


dfString = Pre_processing.dfString

stringency_parameters = ['Facial Coverings',  'Testing Policy',
                'Income Support', #'public_information_campaigns',
                'Cancel Public Events'
                ,'Close Public Transport', 'Stay Home Requirements', 'Workplace Closures'   
                ,'Vaccination Policy']

y_axis_parameters = ['New Cases Per Million', 'New Deaths Per Million']
                    #  "total_vaccinations_per_hundred",'total_tests_per_thousand',
                    #  'hosp_patients_per_million', 'stringency_index']
            
dash.register_page(__name__,
    relative_path='/Polynomial_Regression',
    title='Polynomial Regression',
    name='Polynomial Regression')
    
##################################################################################################### functions:
                    
def rss(y, y_hat):
    RSS = sum(np.power((y - y_hat), 2))
    return RSS


# Calculate Mean Absolute error based on observed (sampled) and model-predicted life expectancy
def MAE(observed_life, predicted_life):
    n = len(observed_life)
    differences = observed_life - predicted_life
    absolute_differences = differences.abs()
    return absolute_differences.sum() / n
    


# function that generates feature matrix containing polynomial with given powers of a single input attribute
def create_feature_matrix(input_attr_col, powers):
    df = pd.DataFrame()
    for i in powers:
        col = "h" + str(i)
        df[col] = np.power(input_attr_col, i)

    return df

# Prediction is a dot product of feature matrix and weight vector
def predict(input_features, weights):
    return input_features.dot(weights)

# For printing ease
def poly_print(weights):
    print(np.poly1d(np.flip(weights, 0)))
    print("")
    
# Find which attributes have non-zero weights, print out their names and weights:
def print_nonzero_weights(all_features, lasso_w):
    i = 0
    result = {}
    for x in all_features:
        if (lasso_w[i] == 0):
            pass
        else:
            result[all_features[i]] = lasso_w[i]
        i = i+1
    
    return result

############################################################################################        

layout = html.Div([
    dbc.Row([
        dbc.Row([
            html.Div(['In this part, we use Polynomial Regression to find the effects of each Stringency Policy individually on the new deaths and new cases in different countries.'],
                style={'font-size': '100%', 'background-color':'#d9ead3'}),

            html.Div(['The best polynomial model from linear to 8th grade is found and shown in the graph below. You can choose to include the points in the graph or remove them.'],
                style={'font-size': '100%', 'background-color':'#d9ead3'}),
            
          ]),  
        
    dbc.Row(html.Br()),
    
    dbc.Row([
        dbc.Col([
            html.Div("Choose if you want to see the data points in the graph or not:"),
        ], width = {"size": 4}),
        
        dbc.Col([
            dcc.RadioItems( ["dont include points", "include points"], "dont include points", id='include_points', 
                            labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                            'padding':'0.5rem 0.80rem',
                            'border-radius':'2rem'},
                            style={'display':'flex', 'font-size': '85%'}, 
                            inputStyle={'cursor':'pointer'},                        
                            ),
        ], width = {"size" : 4})
        ]),     
         
        dbc.Row(html.Br()),
           
        dbc.Row([
            dbc.Col([
            html.Label(['Choose Stringency Factors Below:' +   ' Policies developed by governments to prevent or fight back against the spread of Covid-19'],
                       style={'font-weight': 'bold', 'font-size': '100%'}),], width = {"offset":1})
        ], align="center"),

        dbc.Row([
            dbc.Col([
                dcc.Checklist(stringency_parameters , ["Close Public Transport", "Vaccination Policy"] , id = 'stringency_parameter',
                        
                    labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                    'padding':'0.5rem 1rem',
                    'border-radius':'2rem'},
                    style={'display':'flex' , 'font-size' : '90%'}, 
                    inputStyle={'cursor':'pointer'}, 
                    ), ],
                    )
        ]), 
        ], style={'font-size' : 20} , justify="evenly"),
    
    html.Br(),
    
    dbc.Row ([
        dbc.Col([
            dbc.Row ([
                dbc.Col([
                    html.Label(['Analysis Case:'],style={'font-weight': 'bold', "text-align": "center", 'font-size': '100%'}), 
                    ], width = { 'size': 2, 'offset':1}),
                dbc.Col([           
                    dcc.RadioItems( y_axis_parameters, 'New Cases Per Million', id = 'y_axis',        
                        labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                        'padding':'0.5rem 0.80rem',
                        'border-radius':'2rem'},
                        style={'display':'flex', 'font-size': '85%'}, 
                        inputStyle={'cursor':'pointer'}, 
                    ), 
                ],),
            ],  justify="around"),
        #]),  
        ], width = {"size": 6}),
        
        dbc.Col([
            dbc.Row ([
                dbc.Col([
                    html.Label(['Country:'],style={'font-weight': 'bold', "text-align": "center", 'font-size': '100%'}), 
                ], width = { 'size': 2}),
                dbc.Col([
                    dcc.Dropdown(dfString['Location'].unique(), 'Norway', id = 'country',
                    style={'font-size': '100%', "width":'100%', "background-color": '#A5D6A7'}, ), 
                ], width = { 'size': 5}),
            ], justify="left"),
        ]),
        
        #], width = {"size": 1}),  
          
        ], style={'font-size' : 20, "background-color": '#d9ead3'} , justify="evenly"),           
    # html.Div([
    #     dcc.Graph(id='linear_regression_graph',)
    #     ]),
    
    # html.Div(id = 'linear_coefficients',),
    html.Br(),

    dbc.Row(dbc.Col(html.Div("Polynomial Regression Model For Individual Stringency Factors"), style = {'font-size': 25,'font-weight': 'bold', "text-align": "center", }) ),

    dbc.Row ([
        dbc.Col(dcc.Graph( id='poly_regression_graph', ))
        ]),
    dbc.Row([
        dbc.Col(html.Div()),
    ])
    # html.Div(id = 'poly_coefficients',),
    ], style = {"background-color": '#d9ead3', "height": "100vh"})

# @app.callback(
# @callback(
#     Output('linear_regression_graph', 'figure'),
#     Output('linear_coefficients', 'children'),
#     Input('include_points', 'value'),
#     Input('stringency_parameter', 'value'),
#     Input('y_axis', 'value'),
#     Input('country', 'value'))

# ###################################################################################################### linear:
# def update_graph(include_points, parameters, yaxis_, country):
#     coef_text = ''
#     lr_coef = {}
#     # train_data_country = train_data[train_data['Location'] == country]
#     train_data_country = Pre_processing.find_rolling_df(country, 'train')
#     test_data_country = Pre_processing.find_rolling_df(country, 'test')

#     fig = go.Figure()
#     train_data_lr = train_data_country
    
    
#     #define a color pallete foruse in graphs:
#     col_pal = px.colors.qualitative.Plotly
#     col_pal_iterator = itertools.cycle(col_pal) 
    
#     # lists plot a table of information
#     mae = []
#     rss_linear = []
#     coef = []
#     intercept = []
#     for p in parameters:

#         c = next(col_pal_iterator)
        
#         if include_points == 'include points':
#             fig.add_trace(go.Scatter(x = train_data_country[p] , y = train_data_country[yaxis_], mode = 'markers', name = str('data ' + p), marker = dict(color = c),))
#         linear_regressor = LinearRegression() 
#         lr = linear_regressor.fit(train_data_country[p].values.reshape(-1,1) , train_data_country[yaxis_].values.reshape(-1,1))
#         train_data_lr['temp'] = linear_regressor.predict(train_data_country[p].values.reshape(-1,1)) 

#         # store the coefficients
#         lr_coef['c_' + p] = int(lr.coef_[0])
#         lr_coef['i_' + p] = int(lr.intercept_)
        
#         # calculate values predicted by our linear model on test dataframe
#         predicted_new_covid = lr.intercept_ + lr.coef_[0] * test_data_country[p]
        
#         rss_linear.append(int(rss(test_data_country[p], predicted_new_covid)))
#         mae.append(int(MAE(test_data_country[p], predicted_new_covid)))
#         coef.append(int(lr_coef['c_' + p]))
#         intercept.append(int(lr_coef['i_' + p]))
#         fig.add_trace(go.Scatter(x = train_data_lr[p] , y =  train_data_lr['temp'],  mode='lines', name = str('model ' + p) , line = dict(color = c),))
    
#     info = pd.DataFrame(list(zip(parameters, coef, intercept, rss_linear, mae,)), columns = ["parameter" , "intercept", "linear coefficent", "RSS", "MAE"])
#     coef_text = dash_table.DataTable(info.to_dict('records'), [{"name": str(i), "id": str(i)} for i in info.columns])

#     return fig, coef_text
###################################################################################################### linear finish


# @app.callback(
@callback(
    Output('poly_regression_graph', 'figure'),
    # Output('poly_coefficients', 'children'),
    Input('include_points', 'value'),
    Input('stringency_parameter', 'value'),
    Input('y_axis', 'value'),
    Input('country', 'value'))


def update_graph(include_points, parameters, yaxis_, country):
#################################################################################### Poly:
    poly_weights_dict = {}
    fig = make_subplots(y_title = yaxis_, x_title="Stringency Parameters normalized to a scale of 0 to 3 based on strictness (the higher the more strict)")

    # fig = go.Figure()
    train_data_country = Pre_processing.find_rolling_df(country, 'train')
    test_data_country = Pre_processing.find_rolling_df(country, 'test')
    valid_data_country = Pre_processing.find_rolling_df(country, 'valid')
    
    #define a color pallete foruse in graphs:
    col_pal = px.colors.qualitative.Plotly
    col_pal_iterator = itertools.cycle(col_pal) 
        
    # train_data_country = train_data[train_data['Location'] == country]

    # parameters for table columns
    rss_table = []
    mae = []
    coef_list_names = []
    coef0 = []
    coef1 = []
    coef2 = []
    coef3 = []
    coef4 = []
    coef5 = []
    coef6 = []
    coef7 = []
    
    for p in parameters:
        
        c = next(col_pal_iterator)

        if include_points == "include points":
            fig.add_trace(go.Scatter(x = train_data_country[p] , y = train_data_country[yaxis_], mode = 'markers', name = str('real-life data ' + p), marker = {'color': c}))
        sx = p
        sy = yaxis_
        best_rss = 1000000000000000000000
        best_max_power = 0
        max_power_max = 0
        for max_power in range(1, 8):
            
            #calculate y_prediction using linear regressor
            powers = range(0 , max_power + 1)
            x = create_feature_matrix(train_data_country[sx], powers)
            
            ######HERE:
            y = train_data_country[sy].values
            linear_regressor = LinearRegression(fit_intercept = False).fit(x, y)
            
            #calculate y_validation
            x_Validation = create_feature_matrix(valid_data_country[sx], powers)

            # linear_regressor_validation = LinearRegression(fit_intercept = False).fit(x_Validation, linear_regressor.predict(x_Validation))
            y_valid = linear_regressor.predict(x_Validation)
            
            #calculate RSS of validation
            observed_valid_life = valid_data_country[sy]
            rss_valid = rss(observed_valid_life, y_valid)
            
            #store the coefficients
            am_w1 = linear_regressor.coef_
            am_w0 = linear_regressor.intercept_
            
            poly_weights_dict ["w1_" + str(max_power) + p] = am_w1
            # poly_weights_dict ["w0_" + str(max_power) + p] = am_w0
            #find the best RSS
            if (best_rss > rss_valid):
                best_rss = rss_valid
                best_max_power = max_power
        
        # finding the maximum of the best max power to make columns of table:
        if (best_max_power > max_power_max):
            max_power_max = best_max_power
                              
        # now that we know the best RSS, we find the results
        x_test = create_feature_matrix(test_data_country[sx], range(0, best_max_power+1))
        ########HERE:
        # y_test = test_data[sy].values

        # linear_regressor_test = LinearRegression(fit_intercept = False).fit(x_test, y_test)
        x_pred = create_feature_matrix(train_data_country[sx], range(0, best_max_power+1))

        best_poly_model = LinearRegression(fit_intercept = False).fit(x_pred, train_data_country[sy].values)
        y_pred = best_poly_model.predict(x_pred)
        
        y_pred_test = best_poly_model.predict(x_test)

        observed_valid_life = test_data_country[sy]
        best_poly_rss = rss(test_data_country[sy], y_pred_test)

        poly_weight = poly_weights_dict["w1_" + str(best_max_power) + p]
        # poly_intercept = poly_weights_dict["w0_" + str(best_max_power) + p]
        
        train_data_poly = train_data_country
        train_data_poly["y_pred"] = y_pred
        train_data_poly = train_data_poly.reset_index().sort_values(p)
        
        mae.append(int(MAE(test_data_country[p], y_pred_test)))
        rss_table.append(int(best_poly_rss))
        for k in range(0, max_power_max):
            coef_list_names.append(str(max_power_max) + '^' + str(k))
        
        k = len(poly_weight)
        for i in range (k, 8):
            poly_weight = np.insert(poly_weight ,0,  0)


        coef0.append(int(poly_weight[-1]))
        coef1.append(int(poly_weight[-2]))
        coef2.append(int(poly_weight[-3]))
        coef3.append(int(poly_weight[-4]))
        coef4.append(int(poly_weight[-5]))
        coef5.append(int(poly_weight[-6]))
        coef6.append(int(poly_weight[-7]))
        coef7.append(int(poly_weight[-8]))
        
        fig.add_trace(go.Scatter(x = train_data_poly[p] , y = train_data_poly["y_pred"],
                                 mode='lines', name = str('polymodel ' + p), line = dict(color = c),)) 

        fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
        fig.update_layout(paper_bgcolor= "#d9ead3")
        fig.update_layout(title_font_size = 20, uniformtext_minsize=15, uniformtext_mode    ='hide', )
        
    # info = pd.DataFrame(list(zip(parameters, coef0, coef1, coef2, coef3, coef4, coef5, coef6, coef7, rss_table, mae,)), columns = ["parameter", "power_0", "power_1","power_2","power_3","power_4","power_5","power_6","power_7", "RSS", "MAE"])
    # coef_text = dash_table.DataTable(info.to_dict('records'), [{"name": str(i), "id": str(i),  'type':'numeric'} for i in info.columns])

    return fig#, coef_text

###################################################################################

