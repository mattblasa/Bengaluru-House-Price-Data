#reads the columns.json
import json 
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None 

def get_estimated_price(location, sqft, bhk, bath):
    #if element not found, error is thrown
    try: 
        loc_index = __data_columns.index(location.lower())
    except: 
        loc_index = -1 

    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros() 
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('./artifacts/bangalore_price_model.pickle', 'rb') as f:
        __model = pickle.load(f)
    print('Loading Artifacts....completed')

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
   
    print(get_location_names())