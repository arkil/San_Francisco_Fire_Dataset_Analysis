import pandas as pd
import numpy as np

read_df = pd.read_csv('Fire_Incidents.tsv', sep='\t')

read_df.shape

def delete_columns(col):
    if read_df[col].isnull().sum() > read_df[col].count()/2:
        del read_df[col]
for col in read_df.columns:
    delete_columns(col)
	
	
read_df.head()


df_X1 = read_df.interpolate(method ='nearest') 


def Property_Use_Conv(read_df):
        temp = str(read_df)
        temp = temp.strip()
        #temp = re.sub(r'\d+',temp )
        return temp[:3]
		
df_X1['Action Taken Primary'] = df_X1['Action Taken Primary'].apply(lambda s: Property_Use_Conv(s))
df_X1['Action Taken Primary'] = df_X1['Action Taken Primary'].apply(lambda s: 86 if str(s) == '-' or str(s).startswith('Nan') or str(s).startswith('nan') else s)


df_X1['Action Taken Secondary'] = df_X1['Action Taken Secondary'].apply(lambda s: Property_Use_Conv(s))
df_X1['Action Taken Secondary'] = df_X1['Action Taken Secondary'].apply(lambda s: 86 if str(s) == '-' or str(s).startswith('Nan') or str(s).startswith('nan') else s)


df_X1['Action Taken Other'] = df_X1['Action Taken Other'].apply(lambda s: Property_Use_Conv(s))
df_X1['Action Taken Other'] = df_X1['Action Taken Other'].apply(lambda s: 86 if str(s) == '-' or str(s).startswith('Nan') or str(s).startswith('nan') else s)


df_X1['Detector Alerted Occupants'] = df_X1['Detector Alerted Occupants'].apply(lambda s: Property_Use_Conv(s))
df_X1['Detector Alerted Occupants'] = df_X1['Detector Alerted Occupants'].apply(lambda s: 0 if str(s) == '-' or str(s).startswith('u') or str(s).startswith('nan') else s)

df_X1['Station Area'] = df_X1['Station Area'].apply(lambda s: 40 if str(s).startswith('A') or str(s).startswith('H') or str(s).startswith('O') or str(s).startswith('nan') or str(s) == '-' else s)

df_X1['Action Taken Secondary'] = df_X1['Action Taken Secondary'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 or str(s).find('d') != -1 else s)  
df_X1['Action Taken Primary'] = df_X1['Action Taken Primary'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 or str(s).find('d') != -1 else s)  
df_X1['Action Taken Other'] = df_X1['Action Taken Other'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 or str(s).find('d') != -1 else s)  
df_X1['Detector Alerted Occupants'] = df_X1['Detector Alerted Occupants'].apply(lambda s: str(s)[0] if str(s).find('-') != -1 or str(s).find('*') != -1 or str(s).find('d') != -1 else s)  


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
	
df_X1['Weekday'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_weekday(x))
df_X1['Hour'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_hour(x))
df_X1['Month'] = df_X1['Alarm DtTm'].apply(lambda x: convert_date_to_month(x))


def missing_values():
    temp_dict = dict()
    for i in df_X1.columns:
        if df_X1[i].isnull().sum() > 0: 
            temp_dict[i] = df_X1[i].isnull().sum()
    return temp_dict
	
	
missing_values()

for col in df_X1.columns:
    df_X1[col] = df_X1[col].fillna(method='bfill')

	
df_X1 = df_X1.drop('First Unit On Scene', axis = 1)

df_X1 = df_X1.drop(['Alarm DtTm', 'Arrival DtTm', 'Close DtTm'], axis = 1)



df_X1['Battalion'] = pd.Categorical(df_X1['Battalion'])
one_hot = pd.get_dummies(df_X1['Battalion'],prefix='Battalion')
    #df_X2 = df_X2.drop('Battalion',axis = 1)
    # Join the encoded df
df_X1 = df_X1.join(one_hot)
df_X1['Zipcode'] = pd.Categorical(df_X1['Zipcode'])
one_hot = pd.get_dummies(df_X1['Zipcode'],prefix='Zipcode')
    #df_X2 = df_X2.drop('Battalion',axis = 1)
    # Join the encoded df
df_X1 = df_X1.join(one_hot)
df_feature  = df_X1.drop(['Battalion','Zipcode'],axis=1)


# df_y = df_feature['Suppression Personnel']
# df_feature = df_feature.drop(['Suppression Personnel'],axis=1)


#df_X1_mini = df_feature[:100]


df_y = df_feature['Suppression Personnel']
df_feature = df_feature.drop(['Suppression Personnel'],axis=1)


from scipy.sparse import csr_matrix

def dataframetoCSRmatrix(df):
    nrows = len(df)
    nc = len(df.columns)
    idx = {}
    tid = 0
    nnz = nc * nrows
    
    cols= df.columns
    
    for col in cols:
        df[col] = df[col].apply(str)
        for name in df[col].unique():
            idx[col+name] = tid
            tid += 1
    
    ncols = len(idx)
    
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.int)
    
    i=0
    n=0
    
    for index,row in df.iterrows():
        for j,col in enumerate(cols):
            ind[j+n] = idx[col+row[col]]
            val[j+n] = 1
        ptr[i+1] = ptr[i] + nc
        n += nc
        i += 1
    
    mat = csr_matrix((val,ind,ptr), shape=(nrows,ncols), dtype=np.int)
    mat.sort_indices()   
    
    return mat
    
mat1 = dataframetoCSRmatrix(df_feature)


mat1.shape


from sklearn.model_selection import train_test_split
df_feature_train, df_feature_test, df_y_train, df_y_test = train_test_split(mat1, df_y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestRegressor

rfreg = RandomForestRegressor(n_estimators= 10,max_depth=1, random_state=42)
rfreg.fit(df_feature_train, df_y_train)


predicted_y_rf = rfreg.predict(df_feature_test)


set(predicted_y_rf)


from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error


print("r2_score: ",r2_score(df_y_test, predicted_y_rf))
print("explained_variance_score: ",explained_variance_score(df_y_test, predicted_y_rf))
print("mean_absolute_error: ",mean_absolute_error(df_y_test, predicted_y_rf))
print("mean_squared_error: ",mean_squared_error(df_y_test, predicted_y_rf))
print("median_absolute_error: ",median_absolute_error(df_y_test, predicted_y_rf))
print("mean_squared_log_error: ",mean_squared_log_error(df_y_test, predicted_y_rf))