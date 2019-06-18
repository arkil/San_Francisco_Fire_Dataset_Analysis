#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np


# In[44]:


def preprocessInput(read_df):
    df = pd.read_csv('Fire_Incidents.tsv', sep='\t')
    df = df.drop(['Incident Number','Address','Call Number', 'Arrival DtTm', 'Close DtTm', 'Suppression Units', 'EMS Units', 'EMS Personnel', 'Other Units', 'Other Personnel', 'First Unit On Scene', 'Estimated Property Loss', 
         'Estimated Contents Loss', 'Fire Fatalities', 'Fire Injuries', 'Civilian Fatalities', 'Civilian Injuries', 
         'Mutual Aid', 'Action Taken Secondary', 'Action Taken Other', 'Area of Fire Origin', 
         'Ignition Factor Primary', 'Ignition Factor Secondary', 'Heat Source','Item First Ignited',
         'Human Factors Associated with Ignition', 'Floor of Fire Origin', 'Fire Spread','No Flame Spead',
         'Number of floors with minimum damage','Number of floors with significant damage','Number of floors with heavy damage',
         'Number of floors with extreme damage','Detector Type','Detector Operation','Detector Effectiveness',
         'Detector Failure Reason','Automatic Extinguishing System Present','Automatic Extinguishing Sytem Type',
         'Automatic Extinguishing Sytem Perfomance','Automatic Extinguishing Sytem Failure Reason','Location'], axis=1)
    for col in df.columns:
        delete_columns(df, col)
    df_X = df.interpolate(method ='nearest')
    
    
    df_X['Property Use'] = df_X['Property Use'].apply(lambda s: Property_Use_Conv(s))
    df_X['Property Use'] = df_X['Property Use'].apply(lambda s: 500 if str(s) == 'uuu' or str(s) == 'nnn' or str(s) == 'nan' or str(s) == '-' else s)
    df_X['Property Use'] = df_X['Property Use'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 else s)
    df_X['Primary Situation'] = df_X['Primary Situation'].apply(lambda s: Property_Use_Conv(s))
    df_X['Primary Situation'] = df_X['Primary Situation'].apply(lambda s: 650 if str(s) == 'n/a' or str(s).startswith('cr') or str(s).startswith('y') or str(s) == '-' else s)
    df_X['Primary Situation'] = df_X['Primary Situation'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 else s)  
    df_X['Action Taken Primary'] = df_X['Action Taken Primary'].apply(lambda s: Property_Use_Conv(s))
    df_X['Action Taken Primary'] = df_X['Action Taken Primary'].apply(lambda s: 86 if str(s) == '-' or str(s).startswith('Nan') or str(s).startswith('nan') else s)
    df_X['Detector Alerted Occupants'] = df_X['Detector Alerted Occupants'].apply(lambda s: Property_Use_Conv(s))
    df_X['Detector Alerted Occupants'] = df_X['Detector Alerted Occupants'].apply(lambda s: 0 if str(s) == '-' or str(s).startswith('u') or str(s).startswith('nan') else s)
    df_X['Station Area'] = df_X['Station Area'].apply(lambda s: 40 if str(s).startswith('A') or str(s).startswith('H') or str(s).startswith('O') or str(s).startswith('nan') or str(s) == '-' else s)
    df_X['Detector Alerted Occupants'] = df_X['Detector Alerted Occupants'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 or str(s).find('d') != -1 else s)  
    
    
        
    df_X['Weekday'] = df_X['Alarm DtTm'].apply(lambda x: convert_date_to_weekday(x))
    df_X['Hour'] = df_X['Alarm DtTm'].apply(lambda x: convert_date_to_hour(x))
    df_X['Month'] = df_X['Alarm DtTm'].apply(lambda x: convert_date_to_month(x))
    
    missing_values(df_X)

    for col in df_X.columns:
        df_X[col] = df_X[col].fillna(method='bfill')
    df_X['Battalion'] = pd.Categorical(df_X['Battalion'])
    one_hot = pd.get_dummies(df_X['Battalion'],prefix='Battalion')
        #df_X2 = df_X2.drop('Battalion',axis = 1)
        # Join the encoded df
    df_X = df_X.join(one_hot)
    df_X['Zipcode'] = pd.Categorical(df_X['Zipcode'])
    one_hot = pd.get_dummies(df_X['Zipcode'],prefix='Zipcode')
        #df_X2 = df_X2.drop('Battalion',axis = 1)
        # Join the encoded df
    df_X = df_X.join(one_hot)
    df_feature  = df_X.drop(['Battalion','Zipcode'],axis=1)
    df_feature = df_feature.drop(['Alarm DtTm','City','Incident Date','Neighborhood  District'],axis=1)
    
    return df_feature

    


# In[45]:


# remove columns with more than half null values
def delete_columns(df, col):
    if df[col].isnull().sum() > df[col].count()/2:
        del df[col]
        


# In[46]:


def Property_Use_Conv(df):
    temp = str(df)
    temp = temp.strip()
    #temp = re.sub(r'\d+',temp )
    return temp[:3]

from datetime import datetime

def convert_date_to_weekday(tmstp):
    result = datetime.strptime(tmstp, '%m/%d/%Y %I:%M:%S %p').weekday()
    return result
    
def convert_date_to_hour(tmstp):
    result = datetime.strptime(tmstp, '%m/%d/%Y %I:%M:%S %p').hour
    return result

def convert_date_to_month(tmstp):
    result = datetime.strptime(tmstp, '%m/%d/%Y %I:%M:%S %p').month
    return result


# In[47]:


def missing_values(df_X):
    temp_dict = dict()
    for i in df_X.columns:
        if df_X[i].isnull().sum() > 0: 
            temp_dict[i] = df_X[i].isnull().sum()
    return temp_dict


# In[48]:


def imputeColumnValues(train_cols, test_df):
    listofvals = []
    test_cols = list(test_df)
    for col in train_cols:
        if col in test_cols:
            colval = test_df[col]
            list1= colval.tolist()
            listofvals.append(list1[0])
        else :
            #list1 = [0]
            listofvals.append(0)
    listoflists = []
    listoflists.append(listofvals)
    new_df = pd.DataFrame(listoflists, columns=train_cols)
    return new_df


# In[2]:

def getTrainColumns():
    train_df = pd.read_csv('Fire_Incidents.tsv', sep='\t')
    train_df = preprocessInput(train_df)
    #train_df = train_df.drop('Suppression Personnel')
    train_df = train_df.drop(['Suppression Personnel'],axis=1)
    return list(train_df)


# In[ ]:





# In[ ]:




