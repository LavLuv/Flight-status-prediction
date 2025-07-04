
import sklearn as sk
import numpy as np
import pandas as pd

"""                 Flight status prediction            """

#%%     loading the dataset and preprocessing

flights = pd.read_csv("C:\\Users\\Lav\\Desktop\\Lav\\Datasets\\Numerical datasets\\Flight data (2018-2022)\\raw\\Flights_2022_1.csv")


#%%

# selecting the features to work with

flights = flights[['DayOfWeek', 'Operating_Airline', 'Tail_Number', 'OriginAirportID', 
                   'OriginCityName', 'OriginStateName', 'DestAirportID', 'DestCityName', 
                   'DestStateName', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 
                   'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'WheelsOff', 'TaxiOut', 
                   'WheelsOn', 'TaxiIn', 'ArrTime', 'CRSArrTime', 'ArrDelay', 'ArrDelayMinutes', 
                   'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'Cancelled', 'CancellationCode', 
                   'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Distance', 'DistanceGroup', 
                   'CarrierDelay', 'WeatherDelay', 'SecurityDelay', 'LateAircraftDelay']]

# 38 features

#%%     

# printing characteristics of the data

print(f'-> Total number of data samples = {len(flights)}')
print('')

print('# First 5 rows:-')
print('')

print(flights.head())
print('')

print('# Some statistics of the data:-')
print('')

print(flights.info())
print('')

print(flights.describe())
print('')

# df.columns: lists all the columns of df


#%%

# storing data of cancelled flights in another dataframe

Cancelled_flights = flights[flights['Cancelled'] == 1]

# storing data of operating (non-cancelled) flights in another dataframe

Operating_flights = flights[flights['Cancelled'] == 0]

#%%

# deleting/dropping rows from Operating_flights where 'ArrivalDelayGroups' value is NaN

# i.e., deleting rows where 'ArrivalDelayGroups' has missing values

Operating_flights = Operating_flights.dropna(subset = ['ArrivalDelayGroups'])

#%%

# number of unique aircrafts used in the entire flight dataset, i.e., number of unique tail numbers

# list of unique aircrafts

Unique_aircrafts = flights['Tail_Number'].unique()

print(f'-> Number of unique aircrafts = {len(Unique_aircrafts)}')
print('')

# flights['Tail_Number'].describe()

#%%

"""                 Categorical naive Bayes classifier             """

    
#%%

"""                 Predicting 'ArrivalDelayGroups'                """

# apply for operating flights (Operating_flights) only

input_features = ['DayOfWeek', 'Operating_Airline', 'Tail_Number', 'OriginAirportID', 
                  'OriginCityName', 'OriginStateName', 'DestAirportID', 'DestCityName', 
                  'DestStateName', 'DepTimeBlk', 'DepartureDelayGroups', 'CRSArrTimeBlk']

# 12 input features

target_feature = ['ArrivalDelayGroups']

# these feature columns were further discretized

# 22) CRSArrTime

#%%

# function to discretize 'CRSArrTime' (hourly intervals)

CRSArrTimeBlk = []

for value in Operating_flights['CRSArrTime']:
    value_string = str(value)
    
    if len(value_string) == 4:
        string = value_string[0:2] + '00' + '-' + value_string[0:2] + '59'
        
    elif len(value_string) == 3:
        string = '0' + value_string[0] + '00' + '-' + '0' + value_string[0] + '59'
        
    else:
        string = '0000-0059'
        
    CRSArrTimeBlk.append(string)
   
    
#%%
    
# adding this column to our Operating_flights dataframe

Operating_flights.insert(23, 'CRSArrTimeBlk', CRSArrTimeBlk)
        

#%%

"""         preparing dataframe of the input features     """

df_predict_ArrivalDelayGroups = Operating_flights[input_features]

# changing data type of 'DepartureDelayGroups' column from float64 to int64

df_predict_ArrivalDelayGroups['DepartureDelayGroups'] = df_predict_ArrivalDelayGroups['DepartureDelayGroups'].astype('int64')

print(df_predict_ArrivalDelayGroups.info())
print('')

print(df_predict_ArrivalDelayGroups.describe())
print('')

#%%

"""      using Label Encoding to prepare data for Categorical naive Bayes classifier      """

# as the classifier only works with encoded label numbers

from sklearn.preprocessing import LabelEncoder

# list of columns to encode

columns_to_encode = ['DayOfWeek', 'Operating_Airline', 'Tail_Number', 'OriginAirportID', 
                     'OriginCityName', 'OriginStateName', 'DestAirportID', 'DestCityName', 
                     'DestStateName', 'DepTimeBlk', 'DepartureDelayGroups', 'CRSArrTimeBlk']


# creating multiple objects/instances of sklearn's LabelEncoder, one for each column 

for column in columns_to_encode:
    label_encoder = LabelEncoder()
    df_predict_ArrivalDelayGroups[column] = label_encoder.fit_transform(df_predict_ArrivalDelayGroups[column])
    

#%%

print(df_predict_ArrivalDelayGroups)
print('')

print(df_predict_ArrivalDelayGroups.info())
print('')

print(df_predict_ArrivalDelayGroups.describe())
print('')

print(df_predict_ArrivalDelayGroups.head())
print('')

#%%

ArrivalDelayGroups = Operating_flights['ArrivalDelayGroups']

# converting the data type of ArrivalDelayGroups from float64 to int64

ArrivalDelayGroups = ArrivalDelayGroups.astype('int64')

print(ArrivalDelayGroups)
print('')

print(type(ArrivalDelayGroups))
print('')

print(ArrivalDelayGroups.info())
print('')

print(ArrivalDelayGroups.describe())
print('')

#%%

ArrivalDelayGroupsBlocks = []

for val in ArrivalDelayGroups:
    if val in [-2, -1, 0]:
        delay_class = 0
    elif val in [1, 2, 3]:
        delay_class = 1
    elif val in [4, 5, 6]:
        delay_class = 2
    elif val in [7, 8, 9]:
        delay_class = 3
    else:
        delay_class = 4
        
    ArrivalDelayGroupsBlocks.append(delay_class)


print(ArrivalDelayGroupsBlocks)

#%%

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# here,
# X: a dataframe
# y: a series

X_train, X_test, y_train, y_test = train_test_split(df_predict_ArrivalDelayGroups, 
                                                    ArrivalDelayGroupsBlocks, test_size = 0.2, 
                                                    random_state = 89)

categorical_naive_bayes = CategoricalNB()

categorical_naive_bayes.fit(X_train, y_train)

y_pred = categorical_naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


















#%%

""" Predicting 'TaxiOut' and from that, 'WheelsOff', and from that, 'DepTime' (for departure)  """

1) DayOfWeek
2) Operating_Airline
3) Tail_Number
4) OriginAirportID
5) OriginCityName
6) OriginStateName
7) DestAirportID
8) DestCityName
9) DestStateName
10) CRSDepTime
11) DepTime
12) DepDelay
13) DepDelayMinutes
14) DepDel15
15) DepartureDelayGroups
16) DepTimeBlk
17) WheelsOff
18) TaxiOut
19) WheelsOn
20) TaxiIn
21) ArrTime
22) CRSArrTime
23) ArrDelay
24) ArrDelayMinutes
25) ArrDel15
26) ArrivalDelayGroups
27) ArrTimeBlk
28) Cancelled
29) CancellationCode
30) CRSElapsedTime
31) ActualElapsedTime
32) AirTime
33) Distance
34) DistanceGroup
35) CarrierDelay
36) WeatherDelay
37) SecurityDelay
38) LateAircraftDelay

#%%

""" Predicting 'TaxiIn' and from that, 'WheelsOn', and from that, 'ArrTime' (for arrival)     """

1) DayOfWeek
2) Operating_Airline
3) Tail_Number
4) OriginAirportID
5) OriginCityName
6) OriginStateName
7) DestAirportID
8) DestCityName
9) DestStateName
10) CRSDepTime
11) DepTime
12) DepDelay
13) DepDelayMinutes
14) DepDel15
15) DepartureDelayGroups
16) DepTimeBlk
17) WheelsOff
18) TaxiOut
19) WheelsOn
20) TaxiIn
21) ArrTime
22) CRSArrTime
23) ArrDelay
24) ArrDelayMinutes
25) ArrDel15
26) ArrivalDelayGroups
27) ArrTimeBlk
28) Cancelled
29) CancellationCode
30) CRSElapsedTime
31) ActualElapsedTime
32) AirTime
33) Distance
34) DistanceGroup
35) CarrierDelay
36) WeatherDelay
37) SecurityDelay
38) LateAircraftDelay

#%%





















