
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
                  'DestStateName', 'DepTimeBlk', 'DepartureDelayGroups', 'TaxiOutBlk', 
                  'CRSArrTimeBlk']

# 13 input features

target_feature = ['ArrivalDelayGroups']

# these feature columns were further discretized

# 18) TaxiOut
# 22) CRSArrTime

#%%

# function to discretize 'TaxiOut' (intervals of 3 minutes)

TaxiOutBlk = []

for value in Operating_flights['TaxiOut']:
    if (value / 3) == int(value / 3):
        upper_limit = int(value)
        lower_limit = upper_limit - 2
        string = str(lower_limit) + '-' + str(upper_limit)
    else:
        upper_limit = (int(value / 3) + 1) * 3
        lower_limit = upper_limit - 2
        string = str(lower_limit) + '-' + str(upper_limit)
        
    TaxiOutBlk.append(string)
    

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
    
# adding these 2 columns to our Operating_flights dataframe

Operating_flights.insert(19, 'TaxiOutBlk', TaxiOutBlk)
Operating_flights.insert(24, 'CRSArrTimeBlk', CRSArrTimeBlk)
        

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

# applying Global Label Encoding!!!

# (so that the same entity across any column of the dataframe gets encoded in the same global way)

# to preserve the relationship between entities/objects/features of various columns!!

# list of columns to encode

columns_to_encode = ['DayOfWeek', 'Operating_Airline', 'Tail_Number', 'OriginAirportID', 
                     'OriginCityName', 'OriginStateName', 'DestAirportID', 'DestCityName', 
                     'DestStateName', 'DepTimeBlk', 'DepartureDelayGroups', 'TaxiOutBlk', 
                     'CRSArrTimeBlk']

# The extend() method adds the specified list elements (or any iterables) to the end of the current list

# creating an object/instance of sklearn's LabelEncoder 

label_encoder = LabelEncoder()

unique_entities = []

# all the unique entities across all the columns of the dataframe will be stored in this list

# adding all the unique entities

for column in columns_to_encode:
    unique_entities.extend(df_predict_ArrivalDelayGroups[column].unique())
    
# as there will be recurring entities (encoded similarly tho!) in this list, converting it to a set

unique_entities_set = set(unique_entities)

# now that all the elements are unique, converting the set back to a list

unique_entities = list(unique_entities_set)

# fitting the label encoder on the list of unique entities

label_encoder.fit(unique_entities)

# the encoding (mapping of entities to encoded labels) has been done

# now applying the encoding to all the columns of the dataframe

for column in columns_to_encode:
    df_predict_ArrivalDelayGroups[column] = label_encoder.transform(df_predict_ArrivalDelayGroups[column])


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

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# here,
# X: a dataframe
# y: a series

X_train, X_test, y_train, y_test = train_test_split(df_predict_ArrivalDelayGroups, 
                                                    ArrivalDelayGroups, test_size = 0.2, 
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


















