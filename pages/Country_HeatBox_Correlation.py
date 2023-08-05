# #%%
# # visit http://127.0.0.1:8050/ in your web browser.

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import dash
# from dash import Dash, Input, Output, dash_table, dcc, html, callback
# from . import Pre_processing

# # from Globe import get_dataframe

# # app = Dash(__name__)
# dash.register_page(__name__,
#     relative_path='/Correlations',
#     title='Correlations',
#     name='Correlations Table')


# # test = get_dataframe()
# #Covid_Main = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
# #Covid_Main = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\owid-covid-data.csv")

# # df = pd.DataFrame()
# # df = Covid_Main[["location", "continent", "date", "new_cases_per_million" , "total_cases",
# #                 "new_cases", "total_deaths", "new_deaths", "total_cases_per_million", "new_cases_per_million",
# #                 "total_deaths_per_million", "new_deaths_per_million", "hosp_patients_per_million", "total_tests",
# #                 "total_vaccinations", "new_vaccinations" , "stringency_index", "population_density", "aged_65_older", 
# #                 "gdp_per_capita", "cardiovasc_death_rate", "diabetes_prevalence", "female_smokers" , "male_smokers", 
# #                 "hospital_beds_per_thousand", "life_expectancy", "human_development_index", "population"]]

# # #Change the data from daily to weekly using Sum and Mean:--------------------------------------------------------------------------
# # df ['date'] = pd.to_datetime(df ['date']) - pd.to_timedelta(7, unit='d')

# # dfNorwayWeekSum = df.groupby(pd.Grouper(key='date', freq='W-MON'))['new_cases', 'new_deaths', 'new_cases_per_million',
# # 'new_deaths_per_million', 'new_vaccinations'].sum().reset_index().sort_values('date')

# # dfNorwayWeekMean = df.groupby(pd.Grouper(key='date', freq='W-MON'))["total_cases", "total_deaths",
# #                                                                           "total_vaccinations", "total_cases_per_million",
# #                                                                           "total_tests", "stringency_index", "population_density",
# #                                                                           "aged_65_older", "gdp_per_capita", "cardiovasc_death_rate",
# #                                                                           "diabetes_prevalence", "female_smokers", "male_smokers",
# #                                                                           'hosp_patients_per_million', 'hospital_beds_per_thousand',
# #                                                                           'life_expectancy', 'human_development_index', 'population'
# #                                                                           ].mean().reset_index().sort_values('date')

# # dfNorwayWeekMeanSelected = dfNorwayWeekMean[["total_cases", "total_deaths","total_vaccinations",
# #                                             "total_tests", "stringency_index",
# #                                             "hosp_patients_per_million"]].copy()

# # #dfNorwayWeekSumSelected = dfNorwayWeekSum[['new_cases', 'new_deaths', 'new_cases_per_million',
# # #                                         'new_deaths_per_million', 'new_vaccinations']].copy()
    
# dfWorld = Pre_processing.dfWorld
# dfEurope = dfWorld[dfWorld["continent"] == "Europe"]
# corrEurope = pd.DataFrame(columns = dfEurope['location'].unique())


# # Plotting-------------------------------------------------------------------------------------------------------------------------


# layout = html.Div([
#     html.H4('Correlation matrix of countries'),
    
#     dcc.Dropdown(
#         dfEurope['location'].unique(),
#         'Norway',
#         id='Country'),
    
#     dcc.Graph(id="graph"),
#     html.P("Columns:"),
#     dcc.Checklist(
#         id='Parameter',
#         options = ["total_cases", "total_deaths","total_vaccinations", "total_tests", "stringency_index", "hosp_patients_per_million"],
#         value = ["total_cases", "total_deaths","total_vaccinations", "total_tests", "stringency_index", "hosp_patients_per_million"]
#     ),

# ])

# @callback(
#     Output("graph", "figure"), 
#     Input("Country", "value"),
#     Input("Parameter", "value"))

# def filter_heatmap(CountryName, cols):
#     temp = dfEurope[dfEurope["location"] == CountryName][cols].copy()
#     NorwayCorr = pd.DataFrame(temp.corr())
#     fig = px.imshow(NorwayCorr[cols], text_auto=True, color_continuous_scale='RdGy')
#     return fig


# # if __name__ == '__main__':
# #     app.run_server(debug=True)
    
# #%%
