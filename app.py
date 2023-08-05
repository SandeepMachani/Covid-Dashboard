from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP, ],
          suppress_callback_exceptions=True)

pages = {}
for page in dash.page_registry.values():
    pages[page['name']] = page["relative_path"]

app.layout = html.Div([
	# html.H1('Multi-page app with Dash Pages'),
    dbc.Row([
        
        dbc.Col([
            dbc.Button("Explore", id = "globe_data", external_link = True, href = pages['Global Data'], active=True, 
                       size="lg", className="me-1", color="success" ,style= {"font-weight": "bold"}),
            
        ], width = {"size": 1, "offset": 1}),
        
        dbc.Col([
            dbc.Button("Compare and Find", id = "scatter_plot", external_link = True, href = pages['Scatter Plot'], active=True,
                       size="lg", className="me-1", color="success",style= {"font-weight": "bold"}),
        ], width = {"size": 2, "offset": 1}),
                
         dbc.Col([
            dbc.Button("Predict and Adjust", id = "lasso_regression", external_link = True, href = pages['Lasso Regression'], active=True,
                       size="lg", className="me-1", color="success",style= {"font-weight": "bold"}),
        ], width = {"size": 2, "offset": 1}),
                        
        dbc.Col([
            dbc.Button("Detailed Stringency", id = "poly_regression", external_link = True, href = pages['Polynomial Regression'], active=True,
                       size="lg", className="me-1", color="success",style= {"font-weight": "bold"}),
        ], width = {"size": 2, "offset": 1}),
                
        # dbc.Col(
        #     dcc.Link(f"{'Correlations Table'}", href = pages['Correlations Table']),# width = 2, 
        #     style={'font-family': 'serif', 'font-weight': 'bold' , 'font-size': 30, 'font-color': ''}),
                        
    ],
        style= { 'background-color': "#d9ead3" }
        ),
    dbc.Row(
        html.Br(), style= { 'background-color': "#d9ead3" },
        ),
    
    dbc.Row(dash.page_container),
    
    dbc.Row([
    dbc.Tooltip("Find the general information and visualization of Covid-19 progression using several filters",
                target = "globe_data", autohide=False,
                style = {"font-size":18,},
                placement = "bottom"),
    
    dbc.Tooltip("Access to a bubble plot to compare different parameters in a single week in time between countries. Then, you can see the time-series data and filter the data by stringency index",
                target = "scatter_plot", autohide=False,
                style = {"font-size":18,}, placement = "bottom"),
    
    dbc.Tooltip("Find the most effective government policies for each country and learn how to reduce the Covid-19 cases by stablishing proper policies. Also, there is a prediction plot with various options",
                target = "lasso_regression", autohide=False, style = {"font-size":18}, placement = "bottom"),
   
    dbc.Tooltip("Find the effect of individual government policies using best possible polynomial regression to get a closer look at the effects of individual stringency parameters on the Covid-19 infections",
                target = "poly_regression", autohide=False, style = {"font-size":18}, placement = "bottom"),
    ], justify= "left")
])


if __name__ == '__main__':
	app.run_server(debug=True,
                 dev_tools_ui=False, dev_tools_props_check=False, port = 8050)
 
