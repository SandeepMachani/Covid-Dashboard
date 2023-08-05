import pandas as pd
import os

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# df = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\owid-covid-data.csv")
df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")

#CLEANING DATA FOR SCATTER________________________________________________________________________________________________________________
mask = df["location"] == 'World'
df["continent"][mask] = "World"

#renaming every column:
for s in df.columns:
    df = df.rename({ s : s.replace('_', ' ')}, axis = 'columns')
    
for s in df.columns:
    df = df.rename({ s : s.title()}, axis = 'columns')
    
# Remove anything without any Continent
df.dropna(subset = ["Continent"])
df ['Date'] = pd.to_datetime(df ['Date']) #- pd.to_timedelta(7, unit='d')

agg = {'New Cases': 'sum', 'New Deaths' : 'sum', 'New Cases Per Million' : 'sum',
        'New Deaths Per Million' : 'sum', 'New Vaccinations' : 'sum',
        "Total Cases" : 'mean', "Total Deaths" : 'mean', "Total Deaths Per Million": 'mean',
        "Total Vaccinations" :'mean', "Total Cases Per Million":'mean', 'Total Vaccinations Per Hundred':'mean',
        "Total Tests":'mean', "Stringency Index":'mean', "Population Density":'mean',
        "Aged 65 Older":'mean', "Gdp Per Capita":'mean', "Cardiovasc Death Rate":'mean',
        "Diabetes Prevalence":'mean', "Female Smokers":'mean', "Male Smokers":'mean',
        'Hosp Patients Per Million':'mean', 'Hospital Beds Per Thousand':'mean',
        'Life Expectancy':'mean', 'Human Development Index':'mean', 'Population':'mean', 'Handwashing Facilities': 'mean'}


dfWeek = df.groupby([pd.Grouper(key='Date', freq='W-MON') , 'Iso Code', 'Continent', 'Location']).agg(agg).reset_index().sort_values('Date')

# dfWeekContinent = df.groupby([pd.Grouper(key='Date', freq='W-MON') , 'Continent', 'Iso Code', 'Location'])['New Cases', 'New Deaths', 'New Cases Per Million',
#                                                                                'New Deaths Per Million', 'New Vaccinations'].sum().reset_index().sort_values('Date')

# dfWeekSum['Date Weekly'] = dfWeekSum['Date'].apply(lambda x: str(x))
dfWeek['Date Weekly'] = dfWeek['Date'].apply(lambda x: str(x))

#remove the 00:00:00 from the right of the timestamps string
dfWeek['Date Weekly'] = dfWeek['Date Weekly'].map(lambda x: x.rstrip('00:00:00'))


dfAsia =    dfWeek[dfWeek['Continent'] == 'Asia']
dfEurope =  dfWeek[dfWeek['Continent'] == 'Europe']
dfAfrica =  dfWeek[dfWeek['Continent'] == 'Africa']
dfNA =      dfWeek[dfWeek['Continent'] == 'North America']
dfSA =      dfWeek[dfWeek['Continent'] == 'South America']
dfOceania = dfWeek[dfWeek['Continent'] == 'Oceania']
dfWorld = pd.concat([dfAsia, dfEurope, dfAfrica, dfNA, dfSA, dfOceania])



dfWorld['Week'] = dfWorld['Date'].dt.isocalendar().week
dfWorld['Year'] = dfWorld['Date'].dt.isocalendar().year

# finding the week from the start to the end of the dataframe Date (until 2024)
def Week_Number(df):
    
    if (df['Year'] == 2020):
        return df['Week']
    
    elif (df['Year'] == 2021):
        return df['Week'] + 53 # 2020 had 53 weeks
    
    elif (df['Year'] == 2022):
        return df['Week'] + 53 + 52 # 2021 had 52 weeks
    
    elif (df['Year'] == 2023):
        return df['Week'] + 53 + 52 + 52# 2022 had 52 weeks
    
dfWorld['New Time'] = dfWorld.apply(Week_Number, axis = 1)

dfWorld.reset_index(drop=True, inplace=True)


def get_dataframe():
    df = dfWorld
    return df
#________________________________________________________________________________________________________________

#CLEANING DATA FOR REGRESSION__________________________________________________________________________________


dfString = pd.DataFrame()
    
# Adding the columns: week, year, and new_week(which is the sum of all weeks from the start)
def Add_Columns(Temp):
    #Temp ['Day'] = pd.to_Datetime(Temp ['Day'])                
    Temp['Week'] = Temp['Date'].dt.isocalendar().week
    Temp['Year'] = Temp['Date'].dt.isocalendar().year   
    Temp['New Week'] = Temp.apply(Week_Number, axis = 1)  
    
    return(Temp)

# import the data and clean them using the defined functions
# temp1 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\face-covering-policies-covid (1).csv")

# temp2 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\covid-19-testing-policy.csv")

# temp3 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\income-support-covid.csv")

# temp4 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\public-campaigns-covid.csv")

# temp5 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\public-events-covid.csv")

# temp6 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\public-transport-covid.csv")

# temp7 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\stay-at-home-covid.csv")

# temp8 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\workplace-closures-covid.csv")

# temp9 = pd.read_csv(r"C:\Users\Mashp\Desktop\Ds Project\Stringency data\covid-vaccination-policy.csv")

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Face-covering-policies-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp1 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Covid-19-testing-policy.csv"
full_path = os.path.join(absolute_path, relative_path)

temp2 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Income-support-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp3 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Public-campaigns-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp4 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Public-events-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp5 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Public-transport-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp6 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Stay-at-home-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp7 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Workplace-closures-covid.csv"
full_path = os.path.join(absolute_path, relative_path)

temp8 = pd.read_csv(full_path)

absolute_path = os.path.dirname(__file__)
relative_path = "Stringency data\Covid-vaccination-policy.csv"
full_path = os.path.join(absolute_path, relative_path)

temp9 = pd.read_csv(full_path)

# gather all the data into one dataframe
dfString = pd.DataFrame({'Date': temp9['Day'], 'Location': temp9['Entity'], 'Iso Code': temp9['Code'],
                        'Facial Coverings': temp1['facial_coverings'], 'Testing Policy': temp2['testing_policy'],
                        "Income Support": temp3["income_support"],
# The data for public information campagins is very discrete and rather useless cause its the same over almost 2 years"public_information_campaigns": temp4["public_information_campaigns"],
                        "Cancel Public Events": temp5["cancel_public_events"], "Close Public Transport": temp6["close_public_transport"],
                        "Stay Home Requirements": temp7["stay_home_requirements"], "Workplace Closures": temp8["workplace_closures"],
                        "Vaccination Policy": temp9["vaccination_policy"], })
                        # "New Cases Per Million": dfWorld["New Cases Per Million"], "New Deaths Per Million" : dfWorld["New Deaths Per Million"],
                        # "Total Cases": dfWorld["Total Cases"], "Total Deaths": dfWorld["Total Deaths"]})

dfString ['Date'] = pd.to_datetime(dfString ['Date'])

#Normalize data to a scale of 0 to 3 instaed of different scales:
dfString['Vaccination Policy'] = dfString['Vaccination Policy']*3/5
dfString['Cancel Public Events'] = dfString['Cancel Public Events']*3/2
dfString['Income Support'] = dfString['Income Support']*3/2
dfString['Close Public Transport'] = dfString['Close Public Transport']*3/2


agg = {'Facial Coverings': 'mean' , 'Testing Policy': 'mean',
        "Income Support": 'mean', #"public_information_campaigns": 'mean',
        "Cancel Public Events": 'mean', "Close Public Transport":'mean',
        "Stay Home Requirements": 'mean', "Workplace Closures": 'mean',
        "Vaccination Policy": 'mean'}

# making the Date column into weeks
dfStringWeek = dfString.groupby([pd.Grouper(key='Date', freq='W-MON') , 'Iso Code', 'Location']).agg(agg).reset_index().sort_values('Date')
dfStringWeek = Add_Columns(dfStringWeek)

# importing the data from the other dataframe
def get_cases_country(country):
    
    dfStringWeekCountry = dfStringWeek[dfStringWeek['Location'] == country]
    dftemp = dfWorld[dfWorld['Location'] == country]
    
    dftemp.sort_values("New Time")
    dftemp.reset_index(drop=True, inplace=True)
    
    dfStringWeekCountry.sort_values("New Week")
    dfStringWeekCountry.reset_index(drop=True, inplace=True)
    

    dfStringWeekCountry["New Cases Per Million"] =         dftemp["New Cases Per Million"]
    dfStringWeekCountry["New Deaths Per Million"] =        dftemp["New Deaths Per Million"]
    dfStringWeekCountry["Total Cases Per Million"] =       dftemp["Total Cases Per Million"]
    dfStringWeekCountry["Total Deaths Per Million"] =      dftemp["Total Deaths Per Million"]
 
    dfStringWeekCountry = dfStringWeekCountry.dropna()

    return dfStringWeekCountry


# dfStringR = dfStringWeek

# dfStringWeek = dfStringWeek.dropna()
# dfStringR = dfStringR.dropna()

#________________________________________________________________________________________

#moving average of data from history:
def find_rolling_df(country, type):
    
    dfTemp= get_cases_country(country)

    # dfTemp = dfTemp[dfTemp['Location'] == country] 
    dfTemp.reset_index().sort_values('Date')
    
    #Moving Average of ? months?
    dfTemp['Facial Coverings']                    = dfTemp['Facial Coverings'].rolling(8).mean()
    dfTemp['Testing Policy']                      = dfTemp['Testing Policy'].rolling(8).mean()
    dfTemp['Income Support']                      = dfTemp['Income Support'].rolling(8).mean()
    dfTemp['Cancel Public Events']                = dfTemp['Cancel Public Events'].rolling(8).mean()
    dfTemp['Close Public Transport']              = dfTemp['Close Public Transport'].rolling(8).mean()
    dfTemp['Stay Home Requirements']              = dfTemp['Stay Home Requirements'].rolling(8).mean()
    dfTemp['Workplace Closures']                  = dfTemp['Workplace Closures'].rolling(8).mean()
    dfTemp['Vaccination Policy']                  = dfTemp['Vaccination Policy'].rolling(8).mean()

    # dfTemp['New Cases Per Million']               = dfTemp['New Cases Per Million'].rolling(8).mean()
    # dfTemp['New Deaths Per Million']              = dfTemp['New Deaths Per Million'].rolling(8).mean()
    dfTemp['New Cases Per Million']               = dfTemp['New Cases Per Million']
    dfTemp['New Deaths Per Million']              = dfTemp['New Deaths Per Million']
    
    dfTemp = dfTemp.dropna()
    
    #fix weeks from start to the end as integers:
    dfTemp = Add_Columns(dfTemp)
     
    # train, validation, and test data are made:
    train_data_country = dfTemp.sample(frac=0.6, random_state=32)
    valid_and_test = dfTemp.drop(train_data_country.index)
    valid_data_country = valid_and_test.sample(frac=0.5, random_state=50)
    test_data_country = valid_and_test.drop(valid_data_country.index)
    
    
    if (type == 'train'):
        return train_data_country
    if (type == 'valid'):
        return valid_data_country
    if (type == 'test'):
        return test_data_country
    if (type == 'all'):
        return dfTemp




#find rolling of the predicted data in order to visualize:
def find_rolling_for_prediction(dfx):
    dfTemp= dfx
    
    dfTemp.reset_index().sort_values('New Week')
    
    #Moving Average of ? months?
    dfTemp['Facial Coverings']                    = dfTemp['Facial Coverings'].rolling(8).mean()
    dfTemp['Testing Policy']                      = dfTemp['Testing Policy'].rolling(8).mean()
    dfTemp['Income Support']                      = dfTemp['Income Support'].rolling(8).mean()
    dfTemp['Cancel Public Events']                = dfTemp['Cancel Public Events'].rolling(8).mean()
    dfTemp['Close Public Transport']              = dfTemp['Close Public Transport'].rolling(8).mean()
    dfTemp['Stay Home Requirements']              = dfTemp['Stay Home Requirements'].rolling(8).mean()
    dfTemp['Workplace Closures']                  = dfTemp['Workplace Closures'].rolling(8).mean()
    dfTemp['Vaccination Policy']                  = dfTemp['Vaccination Policy'].rolling(8).mean()

    # dfTemp['New Cases Per Million']               = dfTemp['New Cases Per Million'].rolling(8).mean()
    # dfTemp['New Deaths Per Million']              = dfTemp['New Deaths Per Million'].rolling(8).mean()
    
    dfTemp['New Cases Per Million']               = dfTemp['New Cases Per Million']
    dfTemp['New Deaths Per Million']              = dfTemp['New Deaths Per Million']
    # dfTemp["Total Cases"] = dfTemp["Total Cases"]

    # dfTemp = dfTemp.dropna()
    dfTemp.reset_index(drop=True, inplace=True)
    
    return dfTemp



