#!./env/bin/python
import os
import logging

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import pandas as pd 


FILE_NAME = 'rsu[{num}]_bsm.csv'
FILE_NAME2 = 'rsu[{num}]_blt.csv'
FILE_NAME3 = 'rsu[{num}]_wifi.csv'
FILE_NAME4 = 'rsu[{num}]_tpms.csv'
ANTENNA_NUM = 3

def getting_data(path):
    
    """
    This function merge all the data, sort by rssi(discending) the dataframes and round the timestamp
    """
    decimals = 2
    
    data = []
    for i in range(0, ANTENNA_NUM):
        file_name = f'{path}/{FILE_NAME.format(num=i)}'
        if os.path.exists(file_name) and os.path.isfile(file_name):
            data.append(pd.read_csv(file_name))
        else:
            logging.error(f'File {file_name} not found')
            raise FileNotFoundError
    
    data = pd.concat(data, axis=0, ignore_index=True)
    data.sort_values(by='rssi', ascending=False, inplace=True)
    data['ts_rx'] = data['ts_rx'].round(decimals)

    data2 = []
    for i in range(0, ANTENNA_NUM):
        file_name2 = f'{path}/{FILE_NAME2.format(num=i)}'
        if os.path.exists(file_name2) and os.path.isfile(file_name2):
            data2.append(pd.read_csv(file_name2))
        else:
            logging.error(f'File {file_name2} not found')
            raise FileNotFoundError
    
    data2 = pd.concat(data2, axis=0, ignore_index=True)
    data2.sort_values(by='rssi', ascending=False, inplace=True)
    data2['ts_rx'] = data2['ts_rx'].round(decimals)
    
    data3 = []
    for i in range(0, ANTENNA_NUM):
        file_name3 = f'{path}/{FILE_NAME3.format(num=i)}'
        if os.path.exists(file_name3) and os.path.isfile(file_name3):
            data3.append(pd.read_csv(file_name3))
        else:
            logging.error(f'File {file_name3} not found')
            raise FileNotFoundError
    
    data3 = pd.concat(data3, axis=0, ignore_index=True)
    data3.sort_values(by='rssi', ascending=False, inplace=True)
    data3['ts_rx'] = data3['ts_rx'].round(decimals)
    
    data4 = []
    for i in range(0, ANTENNA_NUM):
        file_name4 = f'{path}/{FILE_NAME4.format(num=i)}'
        if os.path.exists(file_name4) and os.path.isfile(file_name4):
            data4.append(pd.read_csv(file_name4))
        else:
            logging.error(f'File {file_name4} not found')
            raise FileNotFoundError
    
    data4 = pd.concat(data4, axis=0, ignore_index=True)
    data4.sort_values(by='rssi', ascending=False, inplace=True)
    data4['ts_rx'] = data4['ts_rx'].round(decimals)
    
    return data, data2, data3, data4

def filter_data(dataframe, dataframe2, dataframe3, dataframe4):
    
    """
    This function filter duplicates and sort by time the dataframes
    """
    
    dataframe_tmp = dataframe.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe_tmp.index
    dataframe_uniq = dataframe.loc[indici_unici]
    dataframe_uniq.sort_values(by='ts_rx', inplace=True)
    
    dataframe2_tmp = dataframe2.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe2_tmp.index
    dataframe2_uniq = dataframe2.loc[indici_unici]
    dataframe2_uniq.sort_values(by='ts_rx', inplace=True)
    
    dataframe3_tmp = dataframe3.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe3_tmp.index
    dataframe3_uniq = dataframe3.loc[indici_unici]
    dataframe3_uniq.sort_values(by='ts_rx', inplace=True)
    
    dataframe4_tmp = dataframe4.drop(['rsu', 'rssi'], axis=1).drop_duplicates()
    indici_unici = dataframe4_tmp.index
    dataframe4_uniq = dataframe4.loc[indici_unici]
    dataframe4_uniq.sort_values(by='ts_rx', inplace=True)
    
    return dataframe_uniq, dataframe2_uniq, dataframe3_uniq, dataframe4_uniq

def create_csv(dataframe, dataframe2, dataframe3, dataframe4, base_folder, freq, policy):

    """
    This function create all the csv files with the dataframes in directory: '{base_directory}_filt'
    """
    
    path = f'{base_folder}_uniq/fq_{freq}Hz/pc_{policy}' #destination directory
    Path(path).mkdir(parents=True, exist_ok=True)
    
    for r in range(0, ANTENNA_NUM):
        file_name = f'{path}/{FILE_NAME.format(num=r)}'
        dataframe_rsu=dataframe.loc[dataframe['rsu']== r]
        dataframe_rsu.to_csv(file_name, index = False)
        
    for r in range(0, ANTENNA_NUM):
        file_name2 = f'{path}/{FILE_NAME2.format(num=r)}'
        dataframe2_rsu=dataframe2.loc[dataframe2['rsu']== r]
        dataframe2_rsu.to_csv(file_name2, index = False)
        
    for r in range(0, ANTENNA_NUM):
        file_name3 = f'{path}/{FILE_NAME3.format(num=r)}'
        dataframe3_rsu=dataframe3.loc[dataframe3['rsu']== r]
        dataframe3_rsu.to_csv(file_name3, index = False)
        
    for r in range(0, ANTENNA_NUM):
        file_name4 = f'{path}/{FILE_NAME4.format(num=r)}'
        dataframe4_rsu=dataframe4.loc[dataframe4['rsu']== r]
        dataframe4_rsu.to_csv(file_name4, index = False)

def main(base_folder, freq, policy):
    """
    This function call in sequence the other methods
    """
    path = f'{base_folder}/fq_{freq}Hz/pc_{policy}'
    path_if_directory(path)
    
    logging.info('Getting data...')
    dataframe, dataframe2, dataframe3, dataframe4 = getting_data(path)
    logging.info('Filter data...')
    dataframe, dataframe2, dataframe3, dataframe4 = filter_data(dataframe, dataframe2, dataframe3, dataframe4)
    logging.info('Creating csv files...')
    create_csv(dataframe, dataframe2, dataframe3, dataframe4, base_folder, freq, policy)
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
    args = parser.parse_args()

    main(args.directory, args.freq, args.policy)