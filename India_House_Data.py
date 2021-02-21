#!/usr/bin/env python
# coding: utf-8

# dependent variable is price in indian rupees 

# In[2]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plot 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# In[3]:


matplotlib.rcParams['figure.figsize'] =(20,10)


# In[4]:


df1 = pd.read_csv('Bengaluru_House_Data.csv')
df1.head()


# In[5]:


df1.shape


# In[6]:


df1.groupby('area_type')['area_type'].agg('count')


# In[7]:


df2 = df1.drop(['area_type', 'society','balcony','availability'], axis = 'columns')


# In[8]:


df2.head()


# In[9]:


df2.describe()


# In[10]:


df2.info()


# In[11]:


import seaborn as sns 


# # Data Cleaning

# In[12]:


df2.isnull().sum()


# In[13]:


df3 = df2.dropna()
df3.isnull().sum()


# In[14]:


df3.shape


# In[15]:


df3['size'].unique()


# In[16]:


#Split on space, the [0] index means that you are taking the first number from the split string
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[17]:


df3.head()


# In[18]:


df3.info()


# In[19]:


df3['bhk'].unique()


# In[20]:


df3[df3['bhk']>20]


# In[21]:


df3.total_sqft.unique()


# In[22]:


def is_float(x):
    try:
        float(x)
    except:
        return False 
    return True


# In[23]:


#True for is_float method
df3[df3['total_sqft'].apply(is_float)]


# In[24]:


#False for is_float method
df3[~df3['total_sqft'].apply(is_float)].head(20)


# In[25]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 
        


# In[26]:


convert_sqft_to_num('3090 - 5002')


# In[27]:


convert_sqft_to_num('1100Sq. Yards')


# In[28]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)


# In[29]:


df4.head()


# In[30]:


df4.loc[30]


# # Feature Engineering

# In[31]:


df5 = df4.copy()


# In[32]:


df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# In[33]:


df5.head()


# In[34]:


df5.info()


# In[61]:


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


# In[55]:


#find a something where the total square ft per bedroom is not less that 300 sqft
df6 = df5[~(df5['total_sqft']/df5['bhk']<300)]


# In[50]:


df6.shape


# In[54]:


df6.head()


# Price per Square Feet: High and Low

# In[51]:


df6.price_per_sqft.describe()


# # Remove Outliers (Assuming Normal Distribution)

# Filter out everything beyond 1 std deviation 

# In[1]:


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


# In[52]:


df7 = remove_pps_outliers(df6)


# In[53]:


df7.shape


# In[64]:


import matplotlib.pyplot as plt


# In[65]:


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


# In[66]:


plot_scatter_chart(df7, 'Rajaji Nagar')


# In[67]:


plot_scatter_chart(df7, 'Hebbal')


# In[75]:


df7


# In[76]:


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


# In[77]:


df8 = remove_bhk_outliers(df7)


# In[78]:


df8.head(20)


# In[79]:


df8.shape


# In[ ]:


#test

