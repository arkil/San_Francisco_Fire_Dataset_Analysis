#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[141]:


# %matplotlib inline 


# In[142]:


read_df = pd.read_csv('Fire_Incidents.tsv', sep='\t')


# In[3]:


read_df.shape


# In[4]:


read_df.describe()


# In[5]:


read_df['Station Area'].unique()


# In[6]:


read_df.count()


# In[7]:


read_df['Civilian Fatalities'].head(10)


# In[8]:


read_df['Civilian Fatalities'].unique()


# In[9]:


read_df['Civilian Injuries'].head(10)


# In[10]:


read_df['Civilian Injuries'].unique()


# In[11]:


read_df['Mutual Aid'].head(10)


# In[12]:


read_df['Mutual Aid'].count()


# In[13]:


read_df.count()


# In[14]:


read_df.describe()


# In[15]:


read_df['Zipcode'].unique()


# In[ ]:





# In[16]:


# read_df['Zipcode'].isna().sum()


# In[17]:


# read_df['Zipcode'] = read_df['Zipcode'].dropna()


# In[18]:


read_df.shape


# In[19]:


read_df


# In[20]:


df_X = read_df.drop(['Number of Alarms', 'Number of floors with minimum damage', 'Number of floors with significant damage', 'Ignition Cause', 'Number of floors with heavy damage', 'Number of floors with extreme damage', 'Detector Failure Reason', 'Number of Sprinkler Heads Operating', 'Item First Ignited', 'Automatic Extinguishing Sytem Type', 'Automatic Extinguishing Sytem Perfomance', 'Automatic Extinguishing Sytem Failure Reason', 'Supervisor District', 'Neighborhood  District'  ], axis = 1)


# In[21]:


df_X.describe()


# In[22]:


df_X['Detector Operation'].unique()


# In[23]:


df_X['First Unit On Scene'].unique()


# In[24]:


df_X['Action Taken Secondary'].unique()


# In[25]:


df_X.count()


# In[26]:


df_X.shape


# In[27]:


df_X['Fire Spread'].unique()


# In[28]:


df_X['Primary Situation'].unique()


# In[ ]:





# In[29]:


df_X['Action Taken Other'].unique()


# In[30]:


df_X1 = df_X.drop(['Human Factors Associated with Ignition','Arrival DtTm', 'Close DtTm', 'No Flame Spead', 'Mutual Aid','City','Fire Spread','Detector Type','Detector Operation','Detector Effectiveness', 'Primary Situation', 'Other Units'
 ,'Other Personnel', 'Action Taken Other', 'Action Taken Secondary', 'First Unit On Scene', 'Call Number','Station Area' ], axis=1)


# In[31]:


df_X1['Action Taken Primary'].head(10)


# In[32]:


df_X1['Structure Status'].unique()


# In[33]:


df_X1['Structure Type'].unique()


# In[34]:


df_X1['Floor of Fire Origin'].unique()


# In[35]:


df_X1['Automatic Extinguishing System Present'].unique()


# In[36]:


df_X1.shape


# In[37]:


df_X1['Area of Fire Origin'].unique()
#df_aor = df_X1[df_X1['Area of Fire Origin'] == 'nan']


# In[38]:


df_X1['Box'].unique()


# In[149]:


# # Visualisation
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns

# # Configure visualisations
# %matplotlib inline
# mpl.style.use( 'ggplot' )
# sns.set_style( 'white' )
# pylab.rcParams[ 'figure.figsize' ] = 10, 8

# def plot_correlation_map( df ):
#     corr = df.corr()
#     _ , ax = plt.subplots( figsize =( 20 , 20 ) )
#     cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
#     _ = sns.heatmap(
#         corr, 
#         cmap = cmap,
#         square=True, 
#         cbar_kws={ 'shrink' : .9 }, 
#         ax=ax, 
#         annot = True, 
#         annot_kws = { 'fontsize' : 8 }
#     )


# In[150]:


#plot_correlation_map(df_X1)


# In[145]:


df_X1.shape


# In[146]:


df_X1.head(10)


# In[147]:


# plot_correlation_map(read_df)


# In[44]:


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


# In[45]:


df_X1['Weekday'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_weekday(x))


# In[46]:


df_X1['Hour'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_hour(x))


# In[47]:


df_X1['Month'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_month(x))


# In[48]:


df_X2 = df_X1.drop(['Exposure Number','Location', 'Ignition Factor Secondary', 'Heat Source','Floor of Fire Origin','Box' ,'Structure Status','Alarm DtTm', 'Area of Fire Origin', 'Incident Date', 'Incident Number', 'Action Taken Primary','Ignition Factor Primary', 'Structure Type', 'Automatic Extinguishing System Present','Detectors Present','Property Use','Estimated Property Loss','Estimated Contents Loss', 'Detector Alerted Occupants', 'Address'], axis=1)


# In[49]:


df_X2.head(10)


# In[50]:


list(df_X2)


# In[148]:


# plot_correlation_map(df_X2)


# In[52]:


df_X1['Area of Fire Origin'].unique()


# In[53]:


df_X1['Detector Alerted Occupants'].unique()


# In[54]:


df_X1['Property Use'].unique()


# In[55]:


df_X1['Structure Type'].unique()


# In[56]:


df_X1['Structure Status'].unique()


# In[57]:


df_X1['Structure Status'].value_counts()


# In[58]:


df_X1['Structure Type'].value_counts()


# #### Needed in preprocessing to train the model, not needed to test it

# df_X2 = df_X2.drop_duplicates()

# In[60]:



#df_X2 = df_X2.dropna()


# In[61]:


df_X2.head(10)


# In[62]:


df_X2.shape


# In[63]:


df_X2.count()


# In[64]:


df_X2.head()


# In[ ]:





# In[65]:


#df_X2['Estimated Property Loss'].min()


# In[66]:


#df_X2['Estimated Property Loss'] = df_X2['Estimated Property Loss'].apply(lambda x : x if  x > 0 else 0 )


# In[67]:


#df_X2['Estimated Property Loss'].min()


# In[ ]:





# In[68]:


# Create dataset
#imputed = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
#imputed[ 'Estimated Property Loss' ] = df_X2['Estimated Property Loss'].fillna( df_X2['Estimated Property Loss'].mean() )


# In[69]:


#imputed[ 'Estimated Property Loss' ] = df_X2['Estimated Property Loss'].fillna( df_X2['Estimated Property Loss'].mean() )


# In[70]:


#imputed[ 'Estimated Property Loss' ].min()


# In[71]:


#df_X2['Estimated Contents Loss'].min()


# In[72]:


#df_X2['Estimated Contents Loss'] = df_X2['Estimated Contents Loss'].apply(lambda x : x if  x > 0 else 0 )


# In[73]:


#df_X2['Estimated Contents Loss'].min()


# In[74]:


#imputed[ 'Estimated Contents Loss' ] = df_X2['Estimated Contents Loss'].fillna( df_X2['Estimated Contents Loss'].mean() )


# In[75]:


#imputed[ 'Estimated Contents Loss' ]


# In[76]:


#! pip install impyute


# In[77]:


# from impyute.imputation.cs import mice

# # start the MICE training
# imputed_training=mice(df_X2['Estimated Property Loss'].values)


# In[78]:


#df_X2[ 'Detector Alerted Occupants'][:500]


# In[79]:


#imputed[ 'Estimated Property Loss' ] = df_X2['Estimated Property Loss'].fillna( df_X2['Estimated Property Loss']


# In[80]:


# df_X2 = df_X2[df_X2.Zipcode != 'Nan']
# df_X2.where(filter1,inplace=True)


# In[81]:


df_X2 = df_X2[pd.notnull(df_X2['Zipcode'])]


# In[82]:


df_X2.shape


# In[83]:


df_X2.head(10)


# In[ ]:





# In[ ]:





# In[84]:


df_X2['Battalion'].unique()
df_X2['Battalion'] = pd.Categorical(df_X2['Battalion'])
one_hot = pd.get_dummies(df_X2['Battalion'],prefix='Battalion')
#df_X2 = df_X2.drop('Battalion',axis = 1)
# Join the encoded df
df_X2 = df_X2.join(one_hot)
df_X2.head(10)


# In[85]:


df_X2['Zipcode'].unique()
df_X2['Zipcode'] = pd.Categorical(df_X2['Zipcode'])
one_hot = pd.get_dummies(df_X2['Zipcode'],prefix='Zipcode')
#df_X2 = df_X2.drop('Battalion',axis = 1)
# Join the encoded df
df_X2 = df_X2.join(one_hot)


# In[86]:


df_X2.head(10)


# In[87]:


list(df_X2)


# In[88]:


df_y = df_X2['Suppression Personnel']


# In[89]:


df_y


# In[90]:


df_feature  = df_X2.drop(['Suppression Personnel','Battalion','Zipcode'],axis=1)


# In[91]:


df_feature.head(10)


# In[92]:


df_feature.shape


# In[123]:


def preprocessInput(read_df):
    df_X = read_df.drop(['Number of Alarms', 'Number of floors with minimum damage', 'Number of floors with significant damage', 'Ignition Cause', 'Number of floors with heavy damage', 'Number of floors with extreme damage', 'Detector Failure Reason', 'Number of Sprinkler Heads Operating', 'Item First Ignited', 'Automatic Extinguishing Sytem Type', 'Automatic Extinguishing Sytem Perfomance', 'Automatic Extinguishing Sytem Failure Reason', 'Supervisor District', 'Neighborhood  District'  ], axis = 1)
    df_X1 = df_X.drop(['Human Factors Associated with Ignition','Arrival DtTm', 'Close DtTm', 'No Flame Spead', 'Mutual Aid','City','Fire Spread','Detector Type','Detector Operation','Detector Effectiveness', 'Primary Situation', 'Other Units'
 ,'Other Personnel', 'Action Taken Other', 'Action Taken Secondary', 'First Unit On Scene', 'Call Number','Station Area' ], axis=1)
    df_X1['Weekday'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_weekday(x))
    df_X1['Hour'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_hour(x))
    df_X1['Month'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_month(x))
    df_X2 = df_X1.drop(['Exposure Number','Location', 'Ignition Factor Secondary', 'Heat Source','Floor of Fire Origin','Box' ,'Structure Status','Alarm DtTm', 'Area of Fire Origin', 'Incident Date', 'Incident Number', 'Action Taken Primary','Ignition Factor Primary', 'Structure Type', 'Automatic Extinguishing System Present','Detectors Present','Property Use','Estimated Property Loss','Estimated Contents Loss', 'Detector Alerted Occupants', 'Address'], axis=1)
    df_X2['Battalion'].unique()
    df_X2['Battalion'] = pd.Categorical(df_X2['Battalion'])
    one_hot = pd.get_dummies(df_X2['Battalion'],prefix='Battalion')
    #df_X2 = df_X2.drop('Battalion',axis = 1)
    # Join the encoded df
    df_X2 = df_X2.join(one_hot)
    df_X2['Zipcode'] = pd.Categorical(df_X2['Zipcode'])
    one_hot = pd.get_dummies(df_X2['Zipcode'],prefix='Zipcode')
    #df_X2 = df_X2.drop('Battalion',axis = 1)
    # Join the encoded df
    df_X2 = df_X2.join(one_hot)
    df_feature  = df_X2.drop(['Battalion','Zipcode'],axis=1)
    return df_feature


# In[125]:


read_df = pd.read_csv('Fire_Incidents.tsv', sep='\t')
df_feature = preprocessInput(read_df)
df_feature = df_feature.drop_duplicates()
df_y = df_feature['Suppression Personnel']
df_feature = df_feature.drop(['Suppression Personnel'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
df_feature_train, df_feature_test, df_y_train, df_y_test = train_test_split(df_feature, df_y, test_size = 0.2, random_state = 0)


# In[126]:


df_feature_train.shape


# In[127]:


df_feature_test.shape


# In[ ]:





# In[128]:


from sklearn.preprocessing import StandardScaler


# In[129]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(df_feature_train)
X_test = sc_X.fit_transform(df_feature_test)
sc_y = StandardScaler()


# In[130]:


X_train[:10]


# In[131]:


# from sklearn.preprocessing import OneHotEncoder


# In[132]:


# enc = OneHotEncoder()


# In[133]:


# enc.fit(df_feature)


# In[134]:


# onehotlabels = enc.transform(df_feature)


# In[135]:


# onehotlabels.shape


# In[136]:


# from scipy.sparse import csr_matrix

# def dataframetoCSRmatrix(df):
#     nrows = len(df)
#     nc = len(df.columns)
#     idx = {}
#     tid = 0
#     nnz = nc * nrows
    
#     cols= df.columns
    
#     for col in cols:
#         df[col] = df[col].apply(str)
#         for name in df[col].unique():
#             idx[col+name] = tid
#             tid += 1
    
#     ncols = len(idx)
    
#     ind = np.zeros(nnz, dtype=np.int)
#     val = np.zeros(nnz, dtype=np.int)
#     ptr = np.zeros(nrows+1, dtype=np.int)
    
#     i=0
#     n=0
    
#     for index,row in df.iterrows():
#         for j,col in enumerate(cols):
#             ind[j+n] = idx[col+row[col]]
#             val[j+n] = 1
#         ptr[i+1] = ptr[i] + nc
#         n += nc
#         i += 1
    
#     mat = csr_matrix((val,ind,ptr), shape=(nrows,ncols), dtype=np.int)
#     mat.sort_indices()   
    
#     return mat
    


# In[137]:


# mat_train = dataframetoCSRmatrix(df_feature)

# print("Shape of CSR Matrix -", mat.shape)


# In[138]:


# mat_test = dataframetoCSRmatrix(df_feature)


# In[139]:


from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(max_depth=1, random_state=42)
rfclf.fit(X_train, df_y_train)

from sklearn.metrics import f1_score

predicted_y = rfclf.predict(X_test)
print("test accuracy: ",f1_score(df_y_test, predicted_y, average='micro'))


# In[121]:


from sklearn.externals import joblib 
  


# In[122]:


joblib.dump(rfclf, 'sf.pkl') 


# In[ ]:




