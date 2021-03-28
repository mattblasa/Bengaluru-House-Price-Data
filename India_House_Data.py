#!/usr/bin/env python
# coding: utf-8

# dependent variable is price in indian rupees 

# In[1]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plot 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# In[2]:


matplotlib.rcParams['figure.figsize'] =(20,10)


# In[3]:


df1 = pd.read_csv('Bengaluru_House_Data.csv')
df1.head()


# In[4]:


df1.shape


# In[5]:


df1.groupby('area_type')['area_type'].agg('count')


# In[6]:


df2 = df1.drop(['area_type', 'society','balcony','availability'], axis = 'columns')


# In[7]:


df2.head()


# In[8]:


df2.describe()


# In[9]:


df2.info()


# In[10]:


import seaborn as sns 


# # Data Cleaning

# In[11]:


df2.isnull().sum()


# In[12]:


df3 = df2.dropna()
df3.isnull().sum()


# In[13]:


df3.shape


# In[14]:


df3['size'].unique()


# In[15]:


#Split on space, the [0] index means that you are taking the first number from the split string
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[16]:


df3.head()


# In[17]:


df3.info()


# In[18]:


df3['bhk'].unique()


# In[19]:


df3[df3['bhk']>20]


# In[20]:


df3.total_sqft.unique()


# In[21]:


def is_float(x):
    try:
        float(x)
    except:
        return False 
    return True


# In[22]:


#True for is_float method
df3[df3['total_sqft'].apply(is_float)]


# In[23]:


#False for is_float method
df3[~df3['total_sqft'].apply(is_float)].head(20)


# In[24]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 
        


# In[25]:


convert_sqft_to_num('3090 - 5002')


# In[26]:


convert_sqft_to_num('1100Sq. Yards')


# In[27]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)


# In[28]:


df4.head()


# In[29]:


df4.loc[30]


# # Feature Engineering

# In[30]:


df5 = df4.copy()


# In[31]:


df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# In[32]:


df5.head()


# In[33]:


df5.info()


# In[34]:


df5['size'].unique()


# In[35]:


df5.location.unique()


# In[36]:


len(df5.location.unique())


# Dimentionality curse - when you have too many dimensions for feature engineering. 

# In[37]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)


# In[38]:


location_stats


# In[39]:


#how many locations have l
len(location_stats[location_stats<=10])


# In[40]:


#filter a column if the count of the groupby aggregation is less than a number
location_stats_less_than_10 = location_stats[location_stats<=10]


# In[41]:


location_stats_less_than_10


# In[42]:


#reclassifies features as 'other', if name in df5['location'] is in location_stats_less_than_10
df5['location'] = df5['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)


# In[43]:


df5['location']


# In[44]:


len(df5.location.unique())


# In[45]:


df5.head(20)


# In[46]:


df5[df5['location'] == 'other']


# In[47]:


# total_sqr_ft / bedrooms 
#find a something where the total square ft per bedroom is less that 300 sqft
df5[df5['total_sqft']/df5['bhk']<300].head()


# The code above checks for errors. There is no such thing as a 8 bedroom house with 9 baths that is 600 sqft

# In[48]:


df5.shape


# In[49]:


#find a something where the total square ft per bedroom is not less that 300 sqft
df6 = df5[~(df5['total_sqft']/df5['bhk']<300)]


# In[50]:


df6.shape


# In[51]:


df6.head()


# Price per Square Feet: High and Low

# In[52]:


df6.price_per_sqft.describe()


# # Remove Outliers (Assuming Normal Distribution)

# Filter out everything beyond 1 std deviation 

# In[53]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        #mean
        m = np.mean(subdf['price_per_sqft'])
        #standard div of price per sq ft
        st = np.std(subdf['price_per_sqft'])
        #keep data points that are above -1 std and below 1 std
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index =True)
    return df_out


# In[54]:


df7 = remove_pps_outliers(df6)


# In[55]:


df7.shape


# In[56]:


import matplotlib.pyplot as plt


# In[57]:


def plot_scatter_chart(df,location):
    '''This method creates a scatterplot of 2 and 3 bedrooms in a location, and their total area and price per square foot in IND Rupees'''
    #creates two different dataframes: one for 2 bedrooms and 3 bedrooms
    bhk2 = df[(df.location == location) & (df.bhk ==2)]
    bhk3 = df[(df.location == location) & (df.bhk ==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    #create scatterplot
    plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color = 'blue', label='2 BHK', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, marker = '+', color = 'green', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price Per Square Feet')
    plt.title(location)
    plt.legend()


# In[58]:


plot_scatter_chart(df7, 'Rajaji Nagar')


# In[59]:


plot_scatter_chart(df7, 'Hebbal')


# In[60]:


df7


# In[61]:


#13:03
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indeces = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')


# In[62]:


df8 = remove_bhk_outliers(df7)


# In[63]:


df8.head(20)


# In[64]:


df8.shape


# In[65]:


#!jupyter nbconvert --to script India_House_Data.ipynb


# In[66]:


plot_scatter_chart(df7,"Hebbal")


# In[67]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(df8.price_per_sqft, rwidth = 0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')


# In[68]:


df8.bath.unique()


# In[69]:


df8[df8.bath>10]


# In[70]:


plt.hist(df8.bath,rwidth = 0.8)
plt.xlabel("number of bathrooms")
plt.ylabel('Count')


# In[71]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[72]:


df10 = df9.drop(['size', 'price_per_sqft'], axis = 'columns')
df10.head()


# # Machine Learning Model

# In[73]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[74]:


df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis = 'columns')


# In[75]:


df11.head()


# In[76]:


df12 = df11.drop('location', axis = 'columns')


# In[77]:


df12.head()


# In[78]:


X = df12.drop('price', axis = 'columns')
X.head()


# In[79]:


y = df12.price


# In[80]:


y.head()


# In[81]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[82]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[83]:


from sklearn.model_selection import ShuffleSplit 
from sklearn.model_selection import cross_val_score 

cv = ShuffleSplit(n_splits = 5, test_size = 2, random_state = 0)

cross_val_score(LinearRegression(), X, y, cv = cv)


# In[84]:


get_ipython().system('jupyter nbconvert --to script India_House_Data.ipynb')


# ## Hyperparameter Tuning

# In[85]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor 


# In[86]:


def find_best_model_using_GridSearchCV(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize' : [True, False]
            }
            
        },
        'lasso' : {
            'model' : Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
        
    }
    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    for algo_name, config, in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score = False)
        gs.fit(X,y)
        scores.append({
            'model' : algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    
    return pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])


# In[87]:


find_best_model_using_GridSearchCV(X,y)


# In[88]:


X.columns


# In[96]:


X


# In[89]:


#predict price based off locationn, square feet, # of baths, and # of bedrooms
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x= np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return lr_clf.predict([x])[0]


# In[90]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[91]:


predict_price('1st Phase JP Nagar', 1000, 3, 3)


# In[92]:


predict_price('Indira Nagar', 1000, 2, 2)


# In[93]:


predict_price('Indira Nagar', 1000, 3, 3)


# # Export Model

# In[94]:


import pickle
with open ('bangalore_price_model.pickle', 'wb') as f:
    pickle.dump(lr_clf,f)


# In[95]:


import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:




