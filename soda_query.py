import os, requests
import pandas as pd
import json


def get_community_boundaries(save=True):
    
    '''
    creates a request for the SODA API:
        https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Neighborhoods/bbvz-uum9
    
    returns: geojson coordinates for neighborhood boundaries
    '''
    
    if os.path.exists('community_boundaries.pickle'):
        print('loading pickle!')
        community_boundaries = pd.read_pickle('community_boundaries.pickle')
        
    else:
        print('calling API!')
    
        url = 'https://data.cityofchicago.org/resource/igwz-8jzy.json'

        community_boundaries = requests.get(url).json()

        community_boundaries = pd.DataFrame.from_records(community_boundaries)
        
        if save:
            # saving, so you don't have to make the request again :praise:
            community_boundaries.to_pickle('community_boundaries.pickle')
            print('saved to pickle!')
    
    return community_boundaries


def get_tract_boundaries(*args):
    
    '''
    creates a request for the SODA API:
        https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Census-Tracts-2010/5jrd-6zik
    
    returns: geojson coordinates for community tract boundaries
    '''
    
    if os.path.exists('tract_boundaries.pickle'):
        print('loading pickle!')
        tract_boundaries = pd.read_pickle('tract_boundaries.pickle')
        
    else:
        print('calling API!')
        
        url = 'https://data.cityofchicago.org/resource/74p9-q2aq.json'

        tract_boundaries = requests.get(url).json()

        tract_boundaries = pd.DataFrame.from_records(tract_boundaries)
        
        if save:
            # saving, so you don't have to make the request again :praise:
            tract_boundaries.to_pickle('tract_boundaries.pickle')
            print('saved to pickle!')
    
    return tract_boundaries


def get_tract_rides(save=True):
    
    '''
    creates a request to the SODA API:
        https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p
        
    to gather a dataframe grouped by date(day), and census tracts (pickup/dropoff)
    '''
    
    if os.path.exists('tract_df.pickle'):
        print('loading pickle!')
        tract_df = pd.read_pickle('tract_df.pickle')
        
    else:
        print('Calling API!')
        
        url = 'https://data.cityofchicago.org/resource/74p9-q2aq.json'
    
        select = 'date_trunc_ymd(trip_start_timestamp) as date,\
        pickup_census_tract,\
        dropoff_census_tract,\
        SUM(trip_total) as total_fare,\
        AVG(trip_total) as avg_fare,\
        SUM(trip_miles) as total_miles,\
        AVG(trip_miles) as avg_trip_mile,\
        COUNT(trip_id) as rides'

        group = 'date,\
        pickup_census_tract,\
        dropoff_census_tract'

        limit = '1000000'
#         where = 'date > "2018-11-20"'
#         where = "date between '2018-11-20T00:00:00' and '2019-04-31T00:00:00'"
#         where=date between '2015-01-10T12:00:00' and '2015-01-10T14:00:00'

        url = 'https://data.cityofchicago.org/resource/m6dm-c72p.json?'

        params = {'$select': select,
                  '$group': group,
                  '$limit': limit}
#                   '$where': where

        tract_df = requests.get(url, params=params)

        tract_df = pd.DataFrame(tract_df)
        
        if save:
            # saving, so you don't have to make the request again :praise:
            tract_df.to_pickle('tract_df.pickle')
            print('saved to pickle!')
    
    return tract_df


def get_community_rides(save=True):
    '''
    creates a request to the SODA API:
        https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p
        
    to gather a dataframe grouped by date(day/hour), and neighborhoods (pickup)
    '''
    
    if os.path.exists('community_df.pickle'):
        print('loading pickle!')
        community_df = pd.read_pickle('community_df.pickle')
    else:
        print('calling API')
    
        # the inital parameters
        select = 'date_trunc_ymd(trip_start_timestamp) as date,\
        date_extract_hh(trip_start_timestamp) as hour,\
        pickup_community_area,\
        SUM(trip_total) as total_fare,\
        AVG(trip_total) as avg_fare,\
        SUM(trip_miles) as total_miles,\
        AVG(trip_miles) as avg_trip_mile,\
        COUNT(trip_id) as rides'
        
        group = 'date,hour,pickup_community_area'
        limit = '10000000'
        where = 'date > "2018-01-01"'
        url = 'https://data.cityofchicago.org/resource/m6dm-c72p.json?'
        
        # combined parameter dictionary
        params = {'$select': select,
                  '$group': group,
                  '$limit': limit,
                  '$where': where}
    
        # making the request
        community_df = requests.get(url, params=params).json()

        # to pandas
        community_df = pd.DataFrame(community_df)
        
        # cleaning and correcting datatypes
        community_df.hour = community_df.hour.astype('int')
        community_df.rides = community_df.rides.astype('int')
        community_df.date = pd.to_datetime(community_df.date)
        
        if save:
            # saving, so you don't have to make the request again :praise:
            community_df.to_pickle('community_df.pickle')
            print('saved to pickle!')
    
    return community_df

def get_weather(start, end, save=True, key=None, *args):
    
    '''
    *** Must input account key
    
    input: start and end dates (as YYYY-MM-DD strings)
    
    creates a request for each day within your start/end dates to the DarkSky API
    
    output: pandas dataframe of hourly weather for the input dates
    
    * if you have a weather_df pickled in the directory this is running,
        it will load that file instead of calling the API
    '''
    # check to see if weather pickle exists
    if os.path.exists('weather_df.pickle'):
        print('loading pickle!')
        weather_df = pd.read_pickle('weather_df.pickle')
        
    else:
        print('calling Dark Sky API')
    
        date_range = [x.isoformat() for x in pd.date_range(start, end)]

        weather_df = pd.DataFrame()
        
        if key == None:
            key = json.load(open('hidden.json', 'r'))['DSkey']
        lat = '41.8781'
        long = '-87.6298'
        exclude = 'currently, flags'

        for date in date_range:

            url = f'https://api.darksky.net/forecast/{key}/{lat},{long},{date}?exclude={exclude}'

            response = requests.get(url).json()
            response = pd.DataFrame(response['hourly']['data'])
            results = results.append(response, ignore_index=True)

        weather_df['time'] = pd.to_datetime(weather_df['time'], unit='s')
        
        if save:
            # saving, so you don't have to make the request again :praise:
            weather_df.to_pickle('weather_df.pickle')
            print('saved to pickle!')
            
    
    return weather_df


def get_scooters(url=None, key=None, save=True):
    
    '''
    # must input key
    use the Socrata API call to dataset 2kfw-zvte
    return dataframe of individual scooter trips over Chicago 2019 pilot
    '''
    # check to see if weather pickle exists
    if os.path.exists('scooter_df.pickle'):
        print('loading pickle!')
        scooter_df = pd.read_pickle('scooter_df.pickle')
        
    else:
        print('calling API!')
        from sodapy import Socrata #client

        if key == None:
            key = json.load(open('hidden.json', 'r'))['CHI']


        client = Socrata(key['url'],
                         key['key'])

        # First 2000 results, returned as JSON from API / converted to Python list of
        # dictionaries by sodapy.
        results = client.get("2kfw-zvte",
                             limit=712_000)

        # Convert to pandas DataFrame
        scooter_df = pd.DataFrame(results)

        scooter_df.drop(columns=[':@computed_region_bdys_3d7i',
                                 ':@computed_region_vrxf_vc4k'],
                       inplace=True)
        if save:
            # saving, so you don't have to make the request again :praise:
            scooter_df.to_pickle('scooter_df.pickle')
            print('saved to pickle!')
    
    return scooter_df