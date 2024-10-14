#!./env/bin/python
import os
import logging

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import pandas as pd 
import numpy as np
import re
from tqdm import tqdm

FILE_NAME = 'rsu[{num}]_bsm.csv'
FILE_NAME2 = 'rsu[{num}]_wifi.csv'
RSU = [(1306.1, 758.98, 0), 
       (1165.49, 1106.19, 0), 
       (965.25, 444.17, 0)]
ANTENNA_NUM = 3
DECIMALS = 2


def getting_data(path):
    
    """
    This function merge all the data, sort by rssi(discending) the dataframes and round the timestamp
    """
    
    data = []
    for i in range(0, ANTENNA_NUM):
        file_name = f'{path}/{FILE_NAME.format(num=i)}'
        if os.path.exists(file_name) and os.path.isfile(file_name):
            data.append(pd.read_csv(file_name))
        else:
            logging.error(f'File {file_name} not found')
            raise FileNotFoundError
    
    data = pd.concat(data, axis=0, ignore_index=True)
    data.sort_values(by='ts_rx', inplace=True)

    data2 = []
    for i in range(0, ANTENNA_NUM):
        file_name2 = f'{path}/{FILE_NAME2.format(num=i)}'
        if os.path.exists(file_name2) and os.path.isfile(file_name2):
            data2.append(pd.read_csv(file_name2))
        else:
            logging.error(f'File {file_name2} not found')
            raise FileNotFoundError
    
    data2 = pd.concat(data2, axis=0, ignore_index=True)
    data2.sort_values(by='ts_rx', inplace=True)
    
    return data, data2


def get_pos_rsu(rsu):
    pos = np.array(RSU[rsu])
    return pos


def filter_bsm(dataframe, range):
    """
    This function filter by distance from rsu the bsm messages
    """    
    dataframe['distance'] = dataframe.apply(lambda row: np.linalg.norm(get_pos_rsu(row['rsu']) - np.array((row['pos_x'], row['pos_y'], 0))), axis=1)   
    dataframe_filtered = dataframe.query('distance <= @range')    
    dataframe_filtered = dataframe_filtered.drop('distance', axis=1)   
    
    return dataframe_filtered


def getNumber(stringa):
    
    match = re.search(r'\[(\d+)\]', stringa)
    if match:
        return int(match.group(1))
    else:
        return None
    
    
def filter_wifi(dataframe, dataframe_filtered, dataframe2, rg, beacon_interval):
    """
    This function filter by distance from rsu the beacon wifi
    """    
    
    #first filter but remain the pseudo that exit and re-enter in a area 
    wifi_to_keep = np.empty(0)
    dataframe2['realID'] = dataframe2.apply(lambda row: getNumber(row['SSID'])*21+63, axis=1)
    for r in range(0, ANTENNA_NUM):
        dataframe_rsu=dataframe_filtered.loc[dataframe_filtered['rsu']== r]
        dataframe2_rsu=dataframe2.loc[dataframe2['rsu']== r]
        vehicles = np.array(dataframe_rsu['realID'].unique())
        for v in tqdm(vehicles.tolist()):
            vehicle_pseudonyms = np.array(dataframe_rsu.loc[dataframe_rsu['realID'] == v]['pseudonym'].unique()) #all the ps of a realID
            for ps in vehicle_pseudonyms.tolist():
                t_min = dataframe_rsu.loc[dataframe_rsu['pseudonym'] == ps].iloc[0]['ts_rx']
                t_max =dataframe_rsu.loc[dataframe_rsu['pseudonym'] == ps].iloc[-1]['ts_rx']
                dataframe2_filtered = dataframe2_rsu.loc[dataframe2_rsu['realID'] == v]
                dataframe2_filtered = dataframe2_filtered.query('ts_rx >= @t_min - @beacon_interval*0.5').query('ts_rx < @t_max + @beacon_interval*0.5')
                wifi_to_keep = np.append(wifi_to_keep, dataframe2_filtered.index)
    dataframe2 = dataframe2.loc[wifi_to_keep.tolist()]
    
    #print(len(dataframe2))
    
    #erase where i'm sure that the pseudonym exit from the range
    wifi_to_remove = np.empty(0)
    for r in range(0, ANTENNA_NUM):
        dataframe_rsu=dataframe.loc[dataframe['rsu']== r]
        dataframe_filtered_rsu=dataframe_filtered.loc[dataframe_filtered['rsu']== r]
        dataframe2_rsu=dataframe2.loc[dataframe2['rsu']== r]
        
        pseudonyms = np.array(pd.unique(dataframe_filtered_rsu['pseudonym']))
        dataframe_rsu = dataframe_rsu[dataframe_rsu['pseudonym'].isin(pseudonyms)]
        dataframe_rsu = dataframe_rsu.query('distance > @rg')
        vehicles = np.array(pd.unique(dataframe_rsu['realID']))
        
        for v in tqdm(vehicles.tolist()):
            dataframe_vehicle=dataframe_rsu.loc[dataframe_rsu['realID']== v]
            vehicle_pseudonyms = np.array(pd.unique(dataframe_vehicle['pseudonym']))
            dataframe2_vehicle = dataframe2_rsu.loc[dataframe2_rsu['realID'] == v]
            for ps in vehicle_pseudonyms.tolist():
                dataframe_pseudo=dataframe_vehicle.loc[dataframe_vehicle['pseudonym']== ps]
                enter = False
                for row in dataframe_pseudo.itertuples():
                    if enter is False:
                        enter= True
                        time_old = row.ts_rx
                        time_new = row.ts_rx
                        continue
                    time_tmp =row.ts_rx
                    if time_tmp > time_new+0.5*beacon_interval and time_tmp < time_new+1.5*beacon_interval:
                        time_new = time_tmp
                    else:# jump of time
                        dataframe2_tmp = dataframe2_vehicle.query('ts_rx >= @time_old - @beacon_interval*0.5').query('ts_rx < @time_new + @beacon_interval*0.5')
                        if dataframe2_tmp.empty is False:
                            wifi_to_remove = np.append(wifi_to_remove, dataframe2_tmp.index)
                        time_old = time_tmp
                        time_new = time_tmp
                dataframe2_tmp = dataframe2_vehicle.query('ts_rx >= @time_old - @beacon_interval*0.5').query('ts_rx < @time_new + @beacon_interval*0.5')
                if dataframe2_tmp.empty is False:
                    wifi_to_remove = np.append(wifi_to_remove, dataframe2_tmp.index)     
                             
    dataframe2.drop(wifi_to_remove, inplace=True)
    dataframe2 = dataframe2.drop('realID', axis=1)            
          
    return dataframe2


def filter_duplicates(dataframe, dataframe2):
    
    dataframe.sort_values(by='rssi', ascending=False, inplace=True)
    dataframe['ts_rx'] = dataframe['ts_rx'].round(DECIMALS)
    
    dataframe_tmp = dataframe.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe_tmp.index
    dataframe_uniq = dataframe.loc[indici_unici]
    dataframe_uniq.sort_values(by='ts_rx', inplace=True)
    
    dataframe2.sort_values(by='rssi', ascending=False, inplace=True)
    dataframe2['ts_rx'] = dataframe2['ts_rx'].round(DECIMALS)
    dataframe2_tmp = dataframe2.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe2_tmp.index
    dataframe2_uniq = dataframe2.loc[indici_unici]
    dataframe2_uniq.sort_values(by='ts_rx', inplace=True)
    
    return dataframe_uniq, dataframe2_uniq

def create_csv(dataframe, dataframe2, base_folder, freq, policy, rg):

    """
    This function create all the csv files with the dataframes in directory: '{base_folder}_{range}rg'
    """
    
    path = f'{base_folder}_{rg}range/fq_{freq}Hz/pc_{policy}' #destination directory
    Path(path).mkdir(parents=True, exist_ok=True)
    
    for r in range(0, ANTENNA_NUM):
        file_name = f'{path}/{FILE_NAME.format(num=r)}'
        dataframe_rsu=dataframe.loc[dataframe['rsu']== r]
        dataframe_rsu.to_csv(file_name, index = False)
        
    for r in range(0, ANTENNA_NUM):
        file_name2 = f'{path}/{FILE_NAME2.format(num=r)}'
        dataframe2_rsu=dataframe2.loc[dataframe2['rsu']== r]
        dataframe2_rsu.to_csv(file_name2, index = False)
    

def main(base_folder, freq, policy, range):
    """
    This function call in sequence the other methods
    """
    path = f'{base_folder}/fq_{freq}Hz/pc_{policy}'
    path_if_directory(path)
    
    logging.info('Getting data...')
    dataframe, dataframe2= getting_data(path)
    
    logging.info('Filtering BSM...')
    dataframe_filtered= filter_bsm(dataframe, range)
    
    beacon_interval = 1/freq
    logging.info('Filtering WIFI messages...')
    dataframe2_filtered= filter_wifi(dataframe, dataframe_filtered, dataframe2, range, beacon_interval)
    
    logging.info('Filtering duplicates...')
    dataframe_filtered, dataframe2_filtered = filter_duplicates(dataframe_filtered, dataframe2_filtered)
    
    logging.info('Creating csv files...')
    create_csv(dataframe_filtered, dataframe2_filtered, base_folder, freq, policy, range)
    logging.info('Created csv files...')


def path_if_directory(s):
    try:
        p = Path(s)
    except (TypeError, ValueError) as e:
        raise ArgumentTypeError(f"Invalid argument '{s}': '{e}'") from e
    
    if not p.is_dir():
        raise ArgumentTypeError(f"'{s}' is not a valid directory path")
    return p


if __name__ == "__main__":
     
    FORMAT = '\n[%(asctime)s]:[%(levelname)s] %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%d/%m/%y %H:%M:%S:%m')
    
    parser = ArgumentParser()
    parser.add_argument("-dir", "--directory", help="Specify the base directory", required=True, type=path_if_directory)
    parser.add_argument("-fq", "--freq", help="Insert the desired frequency", required=True, type=int, choices=[1, 10])
    parser.add_argument("-pc", "--policy", help="Insert the desired policy", required=True, type=int, choices=[i for i in range(1,6)])
    parser.add_argument("-rg", "--range", help="Insert the desired range", required=True, type=int, choices=[i for i in range(50,901)])
    args = parser.parse_args()

    main(args.directory, args.freq, args.policy, args.range)