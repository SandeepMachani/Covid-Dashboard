from dash import Dash, dcc, html, Input, Output, no_update, dash_table, callback
import dash
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from . import Pre_processing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import itertools
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots


dfString = Pre_processing.dfString


stringency_parameters = ['Facial Coverings',  'Testing Policy',
                'Income Support', #'public_information_campaigns',
                'Cancel Public Events'
                ,'Close Public Transport', 'Stay Home Requirements', 'Workplace Closures'   
                ,'Vaccination Policy']

y_axis_parameters = ['Total Cases Per Million', 'Total Deaths Per Million']
                    #  "total_vaccinations_per_hundred",'total_tests_per_thousand',
                    #  'hosp_patients_per_million', 'stringency_index']

stringency_choices = ["Low Stringency", "Medium Stringency", "High Stringency", "Current Stringency"]

# calculate rss
def rss(y, y_hat):
    RSS = sum(np.power((y - y_hat), 2))
    return RSS


#################################################################################### layout and callbacks:

# app = Dash(__name__) 
dash.register_page(__name__,
    relative_path='/Regression_Lasso',
    title='Lasso Regression',
    name='Lasso Regression')

# app.layout = html.Div([ 
layout = html.Div([
    
    dbc.Row([
    
    html.Br(),

        dbc.Row([
            dbc.Col([
                html.Div(html.P(['In this part, we use Lasso Regression to find the effects of various ',
                                       html.Span("Stringency Policies", id="tooltip-target_stringency_definition",
                                                 style = {'font-weight': "bold"}),
                                       ' on the new deaths and new cases in different countries.'],
                        style={'font-size': '100%', 'background-color':'#d9ead3'})), 
                
                html.Div(html.P(['We introduce the ',
                                html.Span("Reactionary", id="tooltip-target_reactionary_definition", style = {'font-weight': "bold"}),
                                ' and ',
                                html.Span("Preventive", id="tooltip-target_preventive_definition", style = {'font-weight': "bold"}),
                                
                                ' policies. Reactionary policies were taken in action after increased infection of Covid-19 was seen. However, Preventive policies are the policies acquired by the government ahead of time which prevented the spread of infections. These can vary between every country due to several other effects.'],
                        style={'font-size': '100%', 'background-color':'#d9ead3'})), 
                
                html.Div(dcc.Markdown("**Hints on working with the graphs:** You can remove and add back the parameters to the pie charts by clicking on their name. Also, you can zoom in the prediction plot by selecting the portion you need and go back by double clicking.")
                    
                ),
                            
                html.Div([],
                        style={'font-size': '100%', 'background-color':'#d9ead3'}), ]
            ),
        ], align="center"),
        
        
        html.Br(),
         
        ], style={'font-size' : 20} , justify="evenly"),
    
    html.Br(),
    
    dbc.Row ([
        dbc.Col([
            dbc.Row ([
                dbc.Col([
                    html.Label(['Analysis Case:'],style={'font-weight': 'bold', "text-align": "center", 'font-size': '100%'}), 
                    ], width = { 'size': 2, 'offset': 1}),
                dbc.Col([           
                    dcc.RadioItems( y_axis_parameters, 'Total Cases Per Million', id = 'y_axis',        
                        labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                        'padding':'0.5rem 0.80rem',
                        'border-radius':'2rem'},
                        style={'display':'flex', 'font-size': '85%', "margin-left": "10px"}, 
                        
                        inputStyle={'cursor':'pointer'}, 
                    ), 
                ]),
            ],  justify="around"),
        #]),  
        ], width = {"size": 6}),
        
        dbc.Col([
            dbc.Row ([
                dbc.Col([
                    html.Label(['Country:'],style={'font-weight': 'bold',
                                                   'font-size': '100%',
                                                    'padding':'0.2rem 0.50rem','border-radius':'1rem',}), 
                ], width = { 'size': 2}),
                dbc.Col([
                    dcc.Dropdown(dfString["Location"].unique(), 'France', id = 'country',
                    style={'font-size': '100%', "width":'100%', "background-color": '#A5D6A7',
                           "text-align": "center",
                           'padding':'0.2rem 0.5rem','border-radius':'2rem',
                           }, ),    
                ], width = { 'size': 5}),
            ], justify="left"),
        ]),
        #], width = {"size": 1}),  
          
        ], style={'font-size' : 20} , justify="evenly"),
    

    
    dbc.Row ([
        dbc.Col([
            dcc.Graph(id='lasso_regression_graph_all' ),
            ], style={'width': '45%', 'display': 'inline-block'}),
                
        
        dbc.Col([
            dbc.Row ([
                dbc.Col([
                    dcc.Graph(id='lasso_regression_graph_positive')
                ], style={'width': '49%', 'display': 'inline-block'}),
                    
                dbc.Col([
                    dcc.Graph(id='lasso_regression_graph_negative')
                ], style={'width': '49%', 'display': 'inline-block'}),    
            ],),
            
        ], style={ 'width': '49%', 'display': 'inline-block'}),
        
    ]),
    
    dbc.Row([ 
            dbc.Col([
             html.Label("Choose Stringency to Show up on the Prediction Graph Below:", id = "string_label",
                        
                    style={'display':'flex', 'font-size': 20, 'font-weight' : 'bold',
                           'background':'#A5D6A7',  'padding':'0.5rem 0.80rem','border-radius':'1rem'},),
             
            ], width= {"size": 4, "offset": 1}),
            dbc.Col([
             dcc.Checklist(options = stringency_choices , id = 'stringency_parameter', value = ["Current Stringency","High Stringency"],
                            labelStyle={'background':'#A5D6A7',   # style of the <label> that wraps the checkbox input and the option's label
                            'padding':'0.5rem 0.80rem',
                            'border-radius':'2rem'},
                            style={'display':'flex', 'font-size': 20}, 
                            inputStyle={'cursor':'pointer',     "margin-right": "10px",},)
                            # inputStyle= {"margin-right": "10px",})
            ], width= {"size": 7} ),
    ]),
    
    dbc.Row([
        html.Div("Lasso Model's Prediction Plot based on Important Stringency Factors",
                 style={'font-weight': 'bold', "text-align": "center", 'font-size': 25})
    ]),
    
    dbc.Row ([
        dbc.Col([
            dcc.Graph(id = "prediction_graph")
        ], width = {"offset":1}),
    ]),
    
    dbc.Tooltip("We set the 4 most effective parameters found according to the pie chart above to Low, Medium, and High which is from minimum to maximum strictness, respectively",
                target = "string_label", autohide=False, style = {"font-size":18}, placement = "top"),

    # dbc.Tooltip("Low Stringency: ",   
    #             target = "stringency_parameter", autohide=False, style = {"font-size":18}, placement = "top"),
    
    dbc.Tooltip("Include policies acquired by governments to fight against spread of Covid-19 spread"
                ,target = "tooltip-target_stringency_definition",
                autohide=False, style = {"font-size":15}, placement = "bottom"),
    
    dbc.Tooltip("Policies acquired by the governments that reduced the amount of Covid-19 infections. These policies are desirable"
                ,target = "tooltip-target_preventive_definition",
                autohide=False, style = {"font-size":15}, placement = "bottom"),
    
    dbc.Tooltip("Policies acquired by the governments that was more of a reaction to the sudden increase of infections. These policies need to be re-adjusted"
                ,target = "tooltip-target_reactionary_definition",
                autohide=False, style = {"font-size":15}, placement = "bottom"),
    ], style = {"background-color": '#d9ead3'})



################################################################################### lasso:

# @app.callback(
@callback(
    Output('lasso_regression_graph_all', 'figure'),
    Output('lasso_regression_graph_positive', 'figure'),
    Output('lasso_regression_graph_negative', 'figure'),
    Output("prediction_graph", "figure"),
    # Output('best_lasso_model', 'children'),
    # Output('lasso_coefficients', 'children'),
    Input('stringency_parameter', 'value'),
    Input('y_axis', 'value'),
    Input('country', 'value'),)

def update_graph(stringency_chosen_list, yaxis_, country,):
    
    yaxis_plot = yaxis_
    
    if yaxis_ == "Total Cases Per Million":
        yaxis_= "New Cases Per Million"
        
    elif yaxis_ == "Total Deaths Per Million":
        yaxis_ = "New Deaths Per Million"

    all_features = stringency_parameters
    train_data_country = Pre_processing.find_rolling_df(country, 'train')
    valid_data_country = Pre_processing.find_rolling_df(country, 'valid')

    x_pred_lasso = train_data_country[all_features]
    x_valid_lasso = valid_data_country[all_features]
    
    coef = []
    penalties = np.logspace(-3, 0, num= 100)
    w = {}
    least_lasso_rss = 100000000000000000000000000000000000000
    best_lambda = 10
    
    
    for l in penalties:
        #HERE:
        y = train_data_country[yaxis_].values
        lasso = linear_model.Lasso(alpha = l, max_iter=1000, normalize = True).fit(x_pred_lasso, y)

        observed_valid_lasso = valid_data_country[yaxis_]
        y_valid_lasso = lasso.predict(x_valid_lasso)
        
        Lasso_w = lasso.coef_
        lasso_rss = rss(observed_valid_lasso, y_valid_lasso)
        if (lasso_rss < least_lasso_rss):
            best_model = lasso
            least_lasso_rss = lasso_rss
            w_best = Lasso_w
            
    w_all = []
    w_pos = []
    w_neg = []
    feature_pos = []
    feature_neg = []
    colors_pos = {}
    colors_neg = {}
    
    i = 0
    w_sum_all = 0
    
    for p in all_features:
        w_sum_all = w_sum_all+ abs(w_best[i])

    colors = {  'Facial Coverings':         '#636EFA',
                'Testing Policy':           '#EF553B',   
                "Income Support":           '#00CC96', 
                'Cancel Public Events':     '#AB63FA',
                "Close Public Transport":   '#FFA15A',
                "Stay Home Requirements":   '#19D3F3',
                "Workplace Closures" :      '#FF6692', 
                "Vaccination Policy":       '#B6E880',
            }

    for p in all_features:
        coef.append(f'{(w_best[i]/w_sum_all):.2f}')
        w_all.append(abs(w_best[i]))
        if w_best[i]>= 0:
            w_pos.append(w_best[i])
            feature_pos.append(p)
            colors_pos[p] = colors[p]
            
        if w_best[i]< 0:
            w_neg.append(abs(w_best[i]))
            feature_neg.append(p)   
            colors_neg[p] = colors[p]
         
        i += 1
        
    df_pos = pd.DataFrame({"Positive Coefficient" : w_pos, "Reactionary": feature_pos})
    df_neg = pd.DataFrame({"Negative Coefficient" : w_neg, "Preventive": feature_neg})
    
    fig_all = px.pie(values = w_all, names = all_features, title = 'Absolute Value of All Lasso Regression Coefficients',
                                                    color= colors)
    
    fig_all.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
    fig_all.update_layout(title_font_size = 20, uniformtext_minsize=15, uniformtext_mode='hide',)
    fig_all.update_layout(paper_bgcolor= "#d9ead3")


    fig_pos = px.pie(df_pos ,values = "Positive Coefficient", names = "Reactionary", title = 'Reactionary Policies : Positive Coefficients',
                                            color = "Reactionary",    color_discrete_map= colors_pos)

    fig_pos.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
    fig_pos.update_layout(title_font_size = 20, uniformtext_minsize=15, uniformtext_mode='hide',)
    fig_pos.update_layout(paper_bgcolor= "#d9ead3")


    fig_neg = px.pie(df_neg ,values = "Negative Coefficient", names = "Preventive", title = 'Preventive Policies : Negative Coefficients',
                                            color = "Preventive",    color_discrete_map=colors_neg)

    fig_neg.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
    fig_neg.update_layout(title_font_size = 20, uniformtext_minsize=15, uniformtext_mode='hide',)
    fig_neg.update_layout(paper_bgcolor= "#d9ead3")

############################################################################################################### doing prediction
    fig_predict = make_subplots(y_title = yaxis_plot, x_title= "Time in weeks from the start of 2020")
    
    best_lasso_model = best_model
    dfStringWeekCountry = Pre_processing.get_cases_country(country)

    dfStringWeekCountry.sort_values('New Week')
    dfStringWeekCountry.reset_index(drop=True, inplace=True)

    fig_predict.add_trace(go.Scatter(x = dfStringWeekCountry["New Week"], y = dfStringWeekCountry[yaxis_plot],
                                    mode='lines', name = "Previous Data", line = dict(color = "blue"),))
     
    # for each chosen in the checklist go through the process of prediction and add the line in figure:
    for stringency_chosen in stringency_chosen_list:

        df_pre_predict = pd.DataFrame({'New Week': dfStringWeekCountry['New Week'],  'Facial Coverings': dfStringWeekCountry['Facial Coverings'],
                                'Testing Policy': dfStringWeekCountry['Testing Policy'], "Income Support": dfStringWeekCountry["Income Support"],
                                'Cancel Public Events': dfStringWeekCountry["Cancel Public Events"], "Close Public Transport": dfStringWeekCountry["Close Public Transport"],
                                "Stay Home Requirements": dfStringWeekCountry["Stay Home Requirements"], "Workplace Closures" : dfStringWeekCountry["Workplace Closures"],
                                "Vaccination Policy": dfStringWeekCountry["Vaccination Policy"],
                                "New Cases Per Million": dfStringWeekCountry["New Cases Per Million"],"New Deaths Per Million": dfStringWeekCountry["New Deaths Per Million"],
                                "Total Cases Per Million": dfStringWeekCountry["Total Cases Per Million"], "Total Deaths Per Million": dfStringWeekCountry["Total Deaths Per Million"]
                                })

        df_pre_predict.reset_index(drop=True, inplace=True)
        
        temp_df =  df_pre_predict
        temp_df_high = df_pre_predict
    
        features = ['Facial Coverings', 'Testing Policy', "Income Support", 'Cancel Public Events' ,
                    "Close Public Transport" ,"Stay Home Requirements", "Workplace Closures","Vaccination Policy"]
        
        previous_factors = []
        for s in features:
            previous_factors.append(temp_df_high[s][len(temp_df_high.index) - 1])
        
                
        if stringency_chosen == "Low Stringency":
            stringency_decided = [1,1,1,1,1,1,1,1]
            
        elif stringency_chosen == "Medium Stringency":
            stringency_decided = [2,2,2,2,2,2,2,2]
            
        elif stringency_chosen == "High Stringency":
            stringency_decided = [3,3,3,3,3,3,3,3]
            
        elif stringency_chosen == "Current Stringency":
            stringency_decided = previous_factors

        important_factors = N_max_elements(features, w_all, 4)
        
        for i in range (1, 12):
            j = 0
            list_of_row = [last_index_week_high(temp_df_high["New Week"], temp_df_high)]
            for s in features:
                if s in important_factors:
                    list_of_row.append(stringency_decided[j])
                elif s not in important_factors: 
                    list_of_row.append(temp_df_high[s][len(temp_df_high.index) - 1])
            j = j+1
            list_of_row.append(0)
            list_of_row.append(0)
            list_of_row.append(0)
            list_of_row.append(0)


            temp_df_high = add_row(temp_df_high, list_of_row)
            
        
        temp_high = Pre_processing.find_rolling_for_prediction(temp_df_high)

        temp_high = pd.DataFrame({'Facial Coverings':        temp_high['Facial Coverings'], 
                            'Testing Policy':           temp_high['Testing Policy'],
                            "Income Support":           temp_high["Income Support"],
                            'Cancel Public Events':     temp_high["Cancel Public Events"],
                            "Close Public Transport":   temp_high["Close Public Transport"],
                            "Stay Home Requirements":   temp_high["Stay Home Requirements"],
                            "Workplace Closures" :      temp_high["Workplace Closures"],
                            "Vaccination Policy":       temp_high["Vaccination Policy"],})
        
        prediction_input = temp_high.tail(12)
                
        predicted_value = best_lasso_model.predict(prediction_input)
        
        # Filter the data that are negative and make them zero:
        j = 0 
        
        for p in predicted_value:
            if p < 0:
                predicted_value[j] = 0
            j= j + 1
            
        predicted_value_sumed = [0 for i in range (12)]

        j = 0
        for p in predicted_value:
            for k in range (j):
                predicted_value_sumed[j] = predicted_value_sumed[j]+ predicted_value[k] 

            j = j+1
                    
        df_predict_only_high = pd.DataFrame({ 'New Week': temp_df_high['New Week'].tail(12),
                                        yaxis_ : temp_df_high [yaxis_].tail(12) })

        df_predict_only_high [yaxis_] = predicted_value_sumed + dfStringWeekCountry[yaxis_plot][len(dfStringWeekCountry[yaxis_plot])- 1] 
            
        fig_predict.add_trace(go.Scatter(x = df_predict_only_high["New Week"], y = df_predict_only_high[yaxis_],
                                        mode='lines', line = {"dash" : 'dash'}, name = str("Predicted Data for 12 Weeks with " +  str(stringency_chosen)),)) #line = dict(color = "red")
                                            

        fig_predict.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 15)))
        fig_predict.update_layout(title_font_size = 20, uniformtext_minsize=15, uniformtext_mode='hide',)
        fig_predict.update_layout(paper_bgcolor= "#d9ead3")

    # info = pd.DataFrame(list(zip(all_features, coef)), columns = ["parameter", "multivariate coefficient"])
    # coef_text = dash_table.DataTable(info.to_dict('records'), [{"name": str(i), "id": str(i)} for i in info.columns])

    return fig_all, fig_pos, fig_neg, fig_predict


# add a certain list to the last row of a dataframe given:
def add_row(dfTemp, list_of_row):
    dfPredicted = dfTemp
    dfPredicted.loc[len(dfTemp.index)] = list_of_row
    return dfPredicted
    

def last_index_week_high(input_list, df):
    return len(input_list) + df["New Week"][0]

# Function returns N largest elements 
def N_max_elements(all_features, list1, N):
    final_list = {}
    list1_temp = list1[:]
    
    for i in range(0, N):
        max1 = 0
         
        for j in range(len(list1_temp)):    
            if list1_temp[j] >= max1:
                k = j
                max1 = list1_temp[j]
                 
        list1_temp[k] = 0
        final_list[all_features[k]] = max1
                 
    return(final_list)

# if __name__ == '__main__':
# 	app.run_server(debug=True, port = 8040)
