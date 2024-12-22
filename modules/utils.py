import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
from collections import defaultdict
import folium
import requests
import time

def get_city_coordinates(city):
    '''
    Function that retrieves coordinates of a city using the Nomatim OpenStreetMap API
    Inputs:
    - city (str): name of the city (with state)
    Outputs:
    - two integers representing the latitude and longitude of the city coordinates if found, None otherwise
    '''
    # Define url and parameters
    url = 'https://nominatim.openstreetmap.org/search'
    parameters = {
        'q': city,
        'format': 'json',
        'limit': 1
    }

    # Send request to url
    response = requests.get(url, params=parameters, headers={'User-Agent': 'MyGeocoderApp'})

    # If the request was successful and the response can be parsed as json code
    if response.status_code == 200 and response.json():
        # Retrieve the first result in the json response
        data = response.json()[0]
        return pd.to_numeric(data['lat']), pd.to_numeric(data['lon']) # returns the coordinates
    else:
        print('Coordinates not found')

        return None
    

    
def fill_na_coordinates(df):
    '''
    Function that takes the flights dataframe and fills the NaN values of the cities and airports
    with city coordinates using the Nominatim API
    Inputs:
    - df (DataFrame): the dataframe to find coordinates for
    Outputs:
    - None
    '''

    # Row indeces with missing coordinates
    nan_indeces = df[df.isna().any(axis=1)].index

    # Initialize dictionary of found coordinates
    found = defaultdict(tuple)

    # Iterate over the rows with missing values
    for idx in nan_indeces:

        # Case: Org_airport_lat is missing
        if pd.isna(df.loc[idx, 'Org_airport_lat']):
            city = df.loc[idx, 'Origin_city']

            # If we have found the city coordinates previously
            if city in found.keys():
                # Fill in the previously found coordinates
                df.loc[idx, 'Org_airport_lat'] = found[city][0]

            # Otherwise, make a call to the API
            else:
                coordinates = get_city_coordinates(city)
                df.loc[idx, 'Org_airport_lat'] = coordinates[0]
                found[city] = coordinates
                time.sleep(1) # pause for a second

        # Case: Org_airport_long is missing
        if pd.isna(df.loc[idx, 'Org_airport_long']):
            city = df.loc[idx, 'Origin_city']

            # If we have found the city coordinates previously
            if city in found.keys():
                # Fill in the previously found coordinates
                df.loc[idx, 'Org_airport_long'] = found[city][1]

            # Otherwise, make a call to the API
            else:
                coordinates = get_city_coordinates(city)
                df.loc[idx, 'Org_airport_long'] = coordinates[1]
                found[city] = coordinates
                time.sleep(1) # pause for a second

        # Case: Dest_airport_lat is missing
        if pd.isna(df.loc[idx, 'Dest_airport_lat']):
            city = df.loc[idx, 'Origin_city']

            # If we have found the city coordinates previously
            if city in found.keys():
                # Fill in the previously found coordinates
                df.loc[idx, 'Dest_airport_lat'] = found[city][0]

            # Otherwise, make a call to the API
            else:
                coordinates = get_city_coordinates(city)
                df.loc[idx, 'Dest_airport_lat'] = coordinates[0]
                found[city] = coordinates
                time.sleep(1) # pause for a second

        # Case: Dest_airport_long is missing
        if pd.isna(df.loc[idx, 'Dest_airport_long']):
            city = df.loc[idx, 'Origin_city']

            # If we have found the city coordinates previously
            if city in found.keys():
                # Fill in the previously found coordinates
                df.loc[idx, 'Dest_airport_long'] = found[city][1]

            # Otherwise, make a call to the API
            else:
                coordinates = get_city_coordinates(city)
                df.loc[idx, 'Dest_airport_long'] = coordinates[1]
                found[city] = coordinates
                time.sleep(1) # pause for a second


