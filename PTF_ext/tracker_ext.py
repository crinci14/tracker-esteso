#!./env/bin/python
import os
import logging

import re
import math
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import pandas as pd 
import numpy as np
from tqdm import tqdm

#TYPE OF CHOICE
class choice:
    small_alone = 1
    large_alone = 2
    small_multi = 3
    large_multi = 4

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# NUMBER OF EAVESDROPPING ANTENNA 
ANTENNA_NUM = 3
# MEAN DISTANCE BETWEEN TWO CONSECUTIVE BSM
MEAN_DISTANCE = 20 
# POSITION TOLERANCE
POS_TOLERANCE = 25 #25
# HEADING TOLERANCE
ANGLE_TOLERANCE = 45

# NAME OF THE FILE 
FILE_NAME = 'rsu[{num}]_bsm.csv'
FILE_NAME2 = 'rsu[{num}]_wifi.csv'

def getting_data(path):
    """Merge all the data in the \'path\' and calculate the average pseudonyms changes.

    Parameters
    ----------
    path : str, required
        The base \'path\' of the data
    
    Returns
    -------
    data-frame : pandas.DataFrame()
        The pandas DataFrame containing all the bsm read
        
    data2-frame : pandas.DataFrame()
        The pandas DataFrame containing all the wifi messages read;
        add realID column
    
    pseudonyms : np.array
        The numpy array containing the unique pseudonyms

    Raises
    ------
    FileNotFoundError
        If no found in path.
    """

    data = []
    data2 = []
    for i in range(0, ANTENNA_NUM):
        file_name2 = f'{path}/{FILE_NAME2.format(num=i)}'
        file_name = f'{path}/{FILE_NAME.format(num=i)}'
        if os.path.exists(file_name2) and os.path.isfile(file_name2):
            data2.append(pd.read_csv(file_name2))
        else:
            logging.error(f'File {file_name2} not found')
            raise FileNotFoundError
        
        if os.path.exists(file_name) and os.path.isfile(file_name):
            data.append(pd.read_csv(file_name))
        else:
            logging.error(f'File {file_name} not found')
            raise FileNotFoundError
    
    data2 = pd.concat(data2, axis=0, ignore_index=True)
    data2.sort_values(by='ts_rx', inplace=True)
    #add column to compare bsm and wifi messages
    data2['realID'] = data2.apply(lambda row: getNumber(row['SSID'])*21+63, axis=1)
    
    data = pd.concat(data, axis=0, ignore_index=True)
    data.sort_values(by='ts_rx', inplace=True)

    wifi = np.array(pd.unique(data2['BSSID']))
    pseudonyms = np.array(pd.unique(data['pseudonym']))
    wifi_num = len(wifi)
    pseudonyms_num = len(pseudonyms)
    vehicles_num = len(pd.unique(data['realID'].values))

    logging.info(f'TOTAL NUMBER OF VEHICLES FROM WHICH RSU RECEIVED WIFI MESSAGES:  {wifi_num}')
    logging.info(f'TOTAL NUMBER OF VEHICLES FROM WHICH RSU RECEIVED BSM MESSAGES:  {vehicles_num}, TOTAL BSM PSEUDONYMS: {pseudonyms_num}')
    logging.info(f'{bcolors.RED}PSEUDONYMS PER VEHICLE (MEAN): {round((pseudonyms_num/vehicles_num), 2)}{bcolors.RESET}')

    return data, data2, pseudonyms


def pseudonym_change_events(dataframe, pseudonyms):
    """This function perform the following actions:
        - Labeling process of the dataset retrieving the entry and exit events of the pseudonyms.
        - Remove for each pseudonym all the unnecessary BSM between the entry.
        - Calculate the degree angle starting from the heading field of the BSM message
        - Calculate the resultant of the speed vector of the BSM message
        - Add new columns

    Parameters
    ----------
    data-frame : pandas.DataFrame, required
        The pandas data-frame which contains all the bsm eavesdropped by the antennas

    pseudonyms : np.array, required
        The numpy array which contains the unique pseudonyms retrieved from the \'mean_pseudonyms_change\' function
    
    Returns
    -------
    data-frame : pandas.DataFrame()
        The updated pandas DataFrame which include new columns:
        - \'event\' column representing the BSM message classes (entry event -> e, exit event -> x)
        - \'angle\' column representing the BSM message angle starting from the two heading field
        - \'speed\' column representing the resultant speed starting from the two vector component of the BSM fields
        - other columns for count messages and rssi
        
    """
    useless_idx = np.empty(0, dtype=int)
    for i in tqdm(pseudonyms.tolist()):  #show a smart progress meter of the loop
        pseudonyms_events = dataframe.loc[dataframe['pseudonym'] == i]
            
        if len(pseudonyms_events) > 1:
            dataframe.loc[pseudonyms_events.iloc[0].name, 'event'] = 'e' #.name= nominativo riga
            dataframe.loc[pseudonyms_events.iloc[-1].name, 'event'] = 'x'
            useless_idx = np.append(useless_idx, np.delete(pseudonyms_events.index.values, [0, -1], 0))
        else:
            dataframe.loc[pseudonyms_events.iloc[0].name, 'event'] = 'ex'

    previus_dim = len(dataframe)
    dataframe.drop(useless_idx, inplace=True)
    actual_dim = len(dataframe)

    assert previus_dim != actual_dim, 'DATA-FRAME NOT REDUCED'

    dataframe['angle'] = dataframe.apply(lambda row: heading_to_angle(row['heading_x'], row['heading_y']), axis=1)
    dataframe['speed'] = dataframe.apply(lambda row: np.sqrt(row['speed_x']**2 + row['speed_y']**2), axis=1)
    
    return dataframe

def near(value1, value2, tolerance):
    """Based on the tolerance input calculate if two value are near each other.


    Parameters
    ----------
    value1 : double, required
        The first value of the comparison 

    value2 : double, required
        The second value of the comparison 

    tolerance : double, required
        The desired tolerance value
    
    Returns
    -------
    bool
        True if the value1 is between value2 - tolerance and value2 + tolerance
        
    """
    if value1 >= (value2 - tolerance) and value1 <= (value2 + tolerance):
        return True
    else:
        return False
    
def heading_to_angle(x_heading, y_heading):
    """Function which converts the two heading vector components to an angle measured in degrees (0° - 360°)

    Parameters
    ----------
    x_heading : double, required
        The first vector component of the BSM message heading

    y_heading : double, required
        The second vector component of the BSM message heading

    Returns
    -------
    angle : double
        True if the value1 is between value2 - tolerance and value2 + tolerance
        
    """
    det = -y_heading
    dot = x_heading
    angle = math.atan2(det, dot) * 180 / math.pi

    if x_heading >= 0 and y_heading > 0:
        angle = 360 + angle
    elif x_heading < 0 and y_heading >= 0:
        angle = 360 + angle

    return angle

def possible_candidate_found(dataframe, matched_idx, last_seen, results, pseudonyms, to_remove_pseudonyms,sequences):
    """This function add the matched pseudonyms to a remove list to and remove the entry and exit events for this specific spedunym from the dataframe. 

    Parameters
    ----------
    data-frame : pandas.DataFrame, required
        The pandas dataframe which contains all the bsm eavesdropped by the antennas

    matched_idx : pandas.index, required
        The pandas dataframe index for the matched BSM message

    last_seen : pandas.DataFrame, required
        The pandas dataframe of one row which contains all the data of the last exit event

    results : dict, required    
        A dictionary containing the key-value of the TP and the FP numbers.

    pseudonyms : np.array, required
        The numpy array which contains the unique pseudonyms retrieved from the \'mean_pseudonyms_change\' function

    to_remove_pseudonyms : np.array, required
        The numpy array which contains all the matched pseudonyms which have to be removed from the original \'pseudonyms\' list 

    Returns
    -------
        pseudonyms : numpy.array
            The numpy array of the remained pseudonyms 


    """
    old_pseudonym = last_seen['pseudonym']
    new_pseudonym = dataframe.loc[matched_idx, 'pseudonym']
    matched_realID=dataframe.loc[matched_idx, 'realID']
    
    found = False
    #if there is an error, the same pseudonym can be in more than one list
    for list in sequences:
        if old_pseudonym == list[-1]:
            list.append(new_pseudonym)
            found = True
            #break
        
    if found is False:
       sequences.append([old_pseudonym,new_pseudonym])
    
    if matched_realID == last_seen['realID']:
        results['tp'] += 1
    else:
        results['fp'] += 1

    dataframe.drop(dataframe[dataframe['pseudonym'] == old_pseudonym].index, inplace=True)
    to_remove_pseudonyms = np.append(to_remove_pseudonyms, np.where(pseudonyms == old_pseudonym))
    
    return dataframe, to_remove_pseudonyms

def getNumber(stringa):
    
    match = re.search(r'\[(\d+)\]', stringa)
    if match:
        return int(match.group(1))
    else:
        return None

def local_change(dataframe, pseudonyms, beacon_interval, results, dimensions=False):
    """This function for each pseudonym of the \'pseudonyms\' list perform the following actions:
        - Retrieve the last pseudonym sighting corresponding to the exit event (x)
        - Filter the data-frame searching for entry events occurred between the time of the current exit event and the bsm sending interval plus a 
            time_tolerance which is 50% of the bsm sending interval
        - Apply an optional additional filter if the vehicle dimensions is considered
        - Apply the positional filter which consider the position of the exit event (x) as reference position and search for entry events with similar position considering a tolerance of \'POSITION_TOLERANCE\'
        - Apply the heading filter which consider the new angle column of the data-frame of the exit event (x) as reference value and search for entry events with similar heading considering a tolerance of \'ANGLE_TOLERANCE\'
        - If there are some events after the filter process of the previous step, the algorithm calculate the euclidean distance between the exit event (x) and all the entry event of the plausible matches and sort in descending order considering the euclidean distance.
        - Perform the final check using the \'near\' function:
            - if the difference between the speed*time_difference and the calculated euclidean distance is below the \'tolerance\' value of 2 meters -> True Positive
            - otherwise -> False Positive
        - If none of the plausible events match the previous conditions, the closest event in term of euclidean distance is evaluated considering if the distance il below the \'MEAN_DISTANCE\' threshold
        - Remove all the matched and unmatched pseudonyms for the original list of the pseudonyms

    Parameters
    ----------
    data-frame : pandas.DataFrame, required
        The pandas dataframe which contains all the bsm eavesdropped by the antennas

    pseudonyms : np.array, required
        The numpy array which contains the unique pseudonyms retrieved from the \'mean_pseudonyms_change\' function
    
    beacon_interval : double, required
        The value of the sending message interval which correspond to the inverse of the frequency.
    
    
    results : dict, required
        A dictionary containing the key-value of the TP and the FP numbers.
        
    dimensions : bool, optional
        Boolean value which if is True indicating the use of the vehicle size filter
    
    Returns
    -------
        pseudonyms : numpy.array
            The numpy array of the remained pseudonyms 
            
        sequences: list
            Contain the list of connected pseudonyms
    """
    sequences = []
    to_remove_pseudonyms = np.empty(0)
    time_tolerance = beacon_interval * 0.5
    
    if dimensions:
        logging.info('Using vehicles dimensions as filter')
        if not ('length' in dataframe.columns and 'width' in dataframe.columns):
            logging.error('Columns length and width required')
            raise ValueError
        
    for p in tqdm(pseudonyms.tolist()):   
       
        #last_seen = dataframe.loc[(dataframe['pseudonym'] == p) & (dataframe['event'] == 'x')]
        first_seen = dataframe.loc[(dataframe['pseudonym'] == p) & ((dataframe['event'] == 'e') | (dataframe['event'] == 'ex'))]
        last_seen = dataframe.loc[(dataframe['pseudonym'] == p) & ((dataframe['event'] == 'x') | (dataframe['event'] == 'ex'))]
            
        if first_seen.empty:
            continue
            
        if last_seen.empty:
            continue
        
        assert len(first_seen) == 1, f'MULTIPLE ENTRANCES EVENTS FOR PSEUDONYM: {p}, {first_seen}'
        assert len(last_seen) == 1, f'MULTIPLE EXITS EVENTS FOR PSEUDONYM: {p}, {last_seen}'
        
        last_seen = last_seen.iloc[0]
        last_seen_time = last_seen['ts_rx']
        # last_seen_rsu = int(last_seen['rsu'])   
        
        #need to get every message without leak
        possible_match = dataframe.loc[(dataframe['event'] == 'e') | (dataframe['event'] == 'ex')]
        time_interval = possible_match['ts_rx'].between(last_seen_time  + beacon_interval - time_tolerance, last_seen_time + beacon_interval + time_tolerance)
        possible_match = possible_match[time_interval]
        
        if dimensions:
            last_seen_width = last_seen['width']
            last_seen_length = last_seen['length']
            possible_match = possible_match.loc[(possible_match['length'] == last_seen_length) & (possible_match['width'] == last_seen_width)]
        
        filter_pos_x = possible_match['pos_x'].between(last_seen['pos_x'] - POS_TOLERANCE, last_seen['pos_x'] + POS_TOLERANCE)
        filter_pos_y = possible_match['pos_y'].between(last_seen['pos_y'] - POS_TOLERANCE, last_seen['pos_y'] + POS_TOLERANCE)
        possible_match = possible_match[(filter_pos_x) & (filter_pos_y)]

        #V1
        #heading_filter = possible_match['angle'].between(last_seen['angle'] - ANGLE_TOLERANCE, last_seen['angle'] + ANGLE_TOLERANCE)
        #possible_match = possible_match[heading_filter]
        
        #V2
        if last_seen['angle'] >= 360-ANGLE_TOLERANCE:
            heading_filter_1 = possible_match['angle'].between(last_seen['angle'] - ANGLE_TOLERANCE, 360)
            heading_filter_2 = possible_match['angle'].between(0, (last_seen['angle'] + ANGLE_TOLERANCE) % 360)

        elif last_seen['angle'] <= ANGLE_TOLERANCE:
            heading_filter_1 = possible_match['angle'].between(0, last_seen['angle'] + ANGLE_TOLERANCE)
            heading_filter_2 = possible_match['angle'].between((last_seen['angle'] - ANGLE_TOLERANCE) % 360, 360)
        
        else:
            heading_filter_1 = possible_match['angle'].between(last_seen['angle'] - ANGLE_TOLERANCE, last_seen['angle'] + ANGLE_TOLERANCE)
            heading_filter_2 = possible_match['angle'].between(last_seen['angle'] - ANGLE_TOLERANCE, last_seen['angle'] + ANGLE_TOLERANCE)
            
        possible_match = possible_match[(heading_filter_1) | (heading_filter_2)] 
        
        # possible_match = possible_match.loc[dataframe['rsu'] == last_seen_rsu]

        if not possible_match.empty:
            last_pos = np.array((last_seen['pos_x'], last_seen['pos_y'], 0))
            possible_match['distance'] = possible_match.apply(lambda row: np.linalg.norm(last_pos - np.array((row['pos_x'], row['pos_y'], 0))), axis=1)
            possible_match = possible_match.sort_values(by='distance')

            for k in range(len(possible_match)):
                current = possible_match.iloc[k]
                if near(last_seen['speed'] * (current['ts_rx'] - last_seen_time), current['distance'], 2): #tolleranza era 2 
                    matched_idx = possible_match.iloc[k:k+1].index.values.astype(int)[0]
                    dataframe, to_remove_pseudonyms = possible_candidate_found(dataframe, matched_idx, last_seen, results, pseudonyms, to_remove_pseudonyms,sequences)
                    break
            else:
                if possible_match.iloc[0]['distance'] <= MEAN_DISTANCE*beacon_interval:
                    matched_idx = possible_match.iloc[0:1].index.values.astype(int)[0]
                    dataframe, to_remove_pseudonyms = possible_candidate_found(dataframe, matched_idx, last_seen, results, pseudonyms, to_remove_pseudonyms,sequences)
    
    pseudonyms = np.delete(pseudonyms, to_remove_pseudonyms.astype(int))
    return pseudonyms, sequences


def attendance_sum(attendance, attenance_tot):
    for a in attendance:
        if a in attenance_tot:
            attenance_tot[a] += 1
        else:
            attenance_tot[a] = 1


def insert(match, list, BSSID, dataframe, dataframe2, beacon_interval, sequences, to_remove_sequences):
    """insert in list if possible and sort it"""
    
    #times of the new list to insert
    t_new_start = dataframe.loc[(dataframe['pseudonym'] == list[0] )].iloc[0]['ts_rx'] #start time of the list to insert
    t_new_finish = dataframe.loc[(dataframe['pseudonym'] == list[-1] )].iloc[-1]['ts_rx']
    i = 0
    insert = False
    
    if BSSID not in match:
        match[BSSID] = [list]
        insert = True
    else:
        for l in match[BSSID]:
            #tempi delle liste già presenti
            t_old_start = dataframe.loc[(dataframe['pseudonym'] == l[0] ) ].iloc[0]['ts_rx']# start time of this current list
            t_old_finish = dataframe.loc[(dataframe['pseudonym'] == l[-1] ) ].iloc[-1]['ts_rx']

            if t_new_finish + beacon_interval * 0.8 < t_old_start:
                insert = True
                break
            
            if t_new_start - beacon_interval * 0.8 < t_old_finish:
                break   
        else:
            insert = True
            i += 1
            
        if insert:
            match[BSSID].insert(i,list)       
           
        i += 1 
        
    #delete dataframe2 items if insert
    if insert:
        dataframe2.drop(dataframe2[(dataframe2['BSSID'] == BSSID) & (dataframe2['ts_rx'] >= t_new_start) & (dataframe2['ts_rx'] <= t_new_finish)].index, inplace=True)
        index = sequences.index(list)
        to_remove_sequences = np.append(to_remove_sequences, index)
    return dataframe2, to_remove_sequences


def create_array(dataframe_list, beacon_interval, type = 'bsm', first_seen_bsm = 0, last_seen_bsm = 0):
    """ Crea gli array con le stesse dimensioni"""
    
    list = np.empty(0)
    rssi_list=np.array(dataframe_list['rssi'])
    mean_rssi_list = dataframe_list['rssi'].mean()
    count = 0
    inserted = 0   
    
    first_seen_time = dataframe_list.iloc[0]['ts_rx']
    last_seen_time = dataframe_list.iloc[-1]['ts_rx'] 
    
    with open('lista.txt', 'w') as f:
        f.write(f'{first_seen_bsm}\n') 
        f.write(f'{last_seen_bsm}\n') 
        
        f.write(f'{first_seen_time}\n') 
        f.write(f'{last_seen_time}\n\n') 
    
    if type == 'wifi':
        first_seen_wifi = dataframe_list.iloc[0]['ts_rx']
        # +0.001 to round up
        count = round((first_seen_wifi - first_seen_bsm+0.001)/ beacon_interval)
        with open('lista.txt', 'a') as f:
            f.write(f'pre- {count}\n\n') 
        if count < 0:
            count = 0
        list = np.append(list,np.full(count,mean_rssi_list))
        
    enter = False
    for row in dataframe_list.itertuples():
        if enter is False:
            enter= True
            time_old = row.ts_rx
            time_new = row.ts_rx
            continue
        time_tmp =row.ts_rx
        if time_tmp > time_new+0.5*beacon_interval and time_tmp < time_new+1.5*beacon_interval:
            time_new = time_tmp
        else:# jump of time
            count =round((time_new - time_old) / beacon_interval) +1
            list = np.append(list,rssi_list[inserted:inserted+count])
            inserted = inserted + count
            count =round((time_tmp - time_new) / beacon_interval) -1
            if count < 0:
                count = 0
            list = np.append(list,np.full(count,mean_rssi_list))
            time_old = time_tmp
            time_new = time_tmp
        
    count =round((time_new - time_old) / beacon_interval) +1
    list = np.append(list,rssi_list[inserted:inserted+count])
    
    if type == 'wifi':
        last_seen_wifi = dataframe_list.iloc[-1]['ts_rx']
        # -0.001 to round down
        count = round((last_seen_bsm - last_seen_wifi-0.001)/ beacon_interval)
        with open('lista.txt', 'a') as f:
            f.write(f'post- {count}\n\n') 
        if count < 0:
            count = 0
        list = np.append(list,np.full(count,mean_rssi_list))
    
    return list
    
def pearson(dataframe, dataframe2, beacon_interval, type, list):
    """ritorna un insieme di possibii bssid """
    
    correlation_array =np.empty(0)
    possible_bssid_1 = pd.DataFrame()
        
    dataframe_list = dataframe[dataframe['pseudonym'].isin(list)]
    len_bsm_base = len(np.array(dataframe_list['rssi']))
    
    first_seen_time = dataframe_list.iloc[0]['ts_rx']
    last_seen_time = dataframe_list.iloc[-1]['ts_rx'] 

    possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 0.5').query('ts_rx < @last_seen_time + @beacon_interval * 0.5')
    mean_rssi = possible_wifi.groupby(['BSSID']).agg(mean_rssi=('rssi', 'mean')).reset_index()
    vehicles_wifi = np.array(pd.unique(possible_wifi['BSSID']))
    
    
    rssi_bsm = create_array(dataframe_list, beacon_interval)         
    len_rssi_bsm=len(rssi_bsm)
    for v in vehicles_wifi.tolist():
        vehicle_wifi_list = possible_wifi.loc[possible_wifi['BSSID']== v]   
        len_wifi_base = len(np.array(vehicle_wifi_list['rssi']))
        
        #to make faster the code 
        if len_wifi_base < len_bsm_base * 0.4:
            correlation_array = np.append(correlation_array, 0)
            continue
        
        rssi_wifi  = create_array(vehicle_wifi_list, beacon_interval, 'wifi', first_seen_time, last_seen_time) 
        len_rssi_wifi = len(rssi_wifi)
        
        #rare cases, adjust at the end of the list
        if len_rssi_wifi < len_rssi_bsm: 
            mean_rssi_vehicle = mean_rssi.loc[mean_rssi['BSSID'] ==v].iloc[0]['mean_rssi']
            rssi_wifi = np.append(rssi_wifi, np.full(len_rssi_bsm - len_rssi_wifi ,mean_rssi_vehicle))
            len_rssi_wifi = len(rssi_wifi)
            
        if len_rssi_wifi > len_rssi_bsm:
            rssi_wifi = rssi_wifi[:len_rssi_bsm]
            len_rssi_wifi = len(rssi_wifi)
        
        assert len_rssi_wifi == len_rssi_bsm, 'array not equals'    
    
        num_values_bsm = np.array(pd.unique(rssi_bsm))
        num_values_wifi = np.array(pd.unique(rssi_wifi))
        
        if len(num_values_bsm) == 1 or len(num_values_wifi) == 1:# if array is constant 
            correlation_array = np.append(correlation_array, 0)
        else:                
            correlation_matrix = np.corrcoef(rssi_bsm.tolist(), rssi_wifi.tolist())
            pearson_corr = correlation_matrix[0, 1]
            correlation_array = np.append(correlation_array, np.absolute(pearson_corr))
    
    if len(correlation_array) > 0:
        coeff_max = np.max(correlation_array)
        
        if type == choice.small_alone or type == choice.small_multi: #small
            indexes = np.where(correlation_array < coeff_max - 0.01)[0]
        else: #large
            indexes = np.where(correlation_array < coeff_max - 0.02)[0]
        
        possible_bssid_1 = np.delete(vehicles_wifi, indexes) #array dei bssid buoni per pearson
    
    return possible_bssid_1


def time(dataframe, dataframe2, beacon_interval, type, list):
    """euristica basata sul tempo, ritorna un insieme di possibii bssid """
    
    possible_bssid_2 = pd.DataFrame()
    
    dataframe_list = dataframe[dataframe['pseudonym'].isin(list)]    
    first_seen_time = dataframe_list.iloc[0]['ts_rx']
    last_seen_time = dataframe_list.iloc[-1]['ts_rx'] 
    
    if type == choice.small_alone or type == choice.small_multi: #small
        possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 0.5').query('ts_rx < @first_seen_time + @beacon_interval * 0.5')
        start_in = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 2.5').query('ts_rx < @first_seen_time - @beacon_interval * 0.5')
        start_out = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @last_seen_time - @beacon_interval * 0.5').query('ts_rx < @last_seen_time + @beacon_interval * 0.5')
        end_in = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @last_seen_time + @beacon_interval * 0.5').query('ts_rx < @last_seen_time + @beacon_interval * 2.5')
        end_out = possible_wifi['BSSID'].unique()
    else: #large
        possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 0.5').query('ts_rx < @first_seen_time + @beacon_interval * 1.5')
        start_in = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 1.5').query('ts_rx < @first_seen_time - @beacon_interval * 0.5')
        start_out = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @last_seen_time - @beacon_interval * 1.5').query('ts_rx < @last_seen_time + @beacon_interval * 0.5')
        end_in = possible_wifi['BSSID'].unique()
        possible_wifi = dataframe2.query('ts_rx >= @last_seen_time + @beacon_interval * 0.5').query('ts_rx < @last_seen_time + @beacon_interval * 1.5')
        end_out = possible_wifi['BSSID'].unique()
          
    start_tot = np.setdiff1d(start_in, start_out)
    end_tot = np.setdiff1d(end_in, end_out)
    possible_bssid_2 = np.intersect1d(start_tot, end_tot)
    
    return possible_bssid_2


def count_and_rssi(dataframe, dataframe2, beacon_interval, type, list):
    """ 2 euristiche(conteggio e rssi base), ritorna un insieme di possibii bssid """
    
    length = len(list)       
    attenance_tot ={} #to check if the wifi messages are present in every pseudonym time
    possible_bssid_3 = pd.DataFrame()
    merge = pd.DataFrame() 
    
    for p in list:
        dataframe_pseudo = dataframe.loc[(dataframe['pseudonym'] == p)]
        if dataframe_pseudo.empty:
            continue
        dataframe_pseudo_values = dataframe_pseudo.agg(count=('ts_rx', 'count'), mean_rssi=('rssi', 'mean'), max_rssi=('rssi', 'max'), min_rssi=('rssi', 'min'), std_rssi=('rssi', 'std'))
        first_seen_time = dataframe_pseudo.iloc[0]['ts_rx']
        last_seen_time = dataframe_pseudo.iloc[-1]['ts_rx']
    
        possible_wifi = dataframe2.query('ts_rx >= @first_seen_time - @beacon_interval * 0.5').query('ts_rx < @last_seen_time + @beacon_interval * 0.5')
        attendance = possible_wifi['BSSID'].unique()
        attendance_sum(attendance, attenance_tot)
    
        #add columns
        if COUNT & RSSI_BASE:
            results = possible_wifi.groupby(['SSID','BSSID']).agg(count=('ts_rx', 'count'), mean_rssi=('rssi', 'mean'), max_rssi=('rssi', 'max'), min_rssi=('rssi', 'min'), std_rssi=('rssi', 'std')).reset_index()
        else:
            if COUNT:
                results = possible_wifi.groupby(['SSID','BSSID']).agg(count=('ts_rx', 'count')).reset_index()
            if RSSI_BASE:
                results = possible_wifi.groupby(['SSID','BSSID']).agg(mean_rssi=('rssi', 'mean'), max_rssi=('rssi', 'max'), min_rssi=('rssi', 'min'), std_rssi=('rssi', 'std')).reset_index()        
        
        if COUNT:
            bsm_count = dataframe_pseudo_values.loc['count', 'ts_rx']
            
        if RSSI_BASE:
            bsm_max_rssi = dataframe_pseudo_values.loc['max_rssi','rssi']
            bsm_ts_max_rssi = dataframe_pseudo.loc[dataframe_pseudo['rssi'] == bsm_max_rssi]['ts_rx'].iloc[0]
            bsm_min_rssi = dataframe_pseudo_values.loc['min_rssi','rssi']
            bsm_ts_min_rssi = dataframe_pseudo.loc[dataframe_pseudo['rssi'] == bsm_min_rssi]['ts_rx'].iloc[0]
            bsm_mean_rssi = dataframe_pseudo_values.loc['mean_rssi','rssi']
            bsm_std_rssi = dataframe_pseudo_values.loc['std_rssi','rssi']
            
            idx = possible_wifi.groupby(['SSID','BSSID'])['rssi'].idxmax()
            results['ts_max_rssi'] = possible_wifi.loc[idx, 'ts_rx'].values
            idx = possible_wifi.groupby(['SSID','BSSID'])['rssi'].idxmin()
            results['ts_min_rssi'] = possible_wifi.loc[idx, 'ts_rx'].values
        
        if type == choice.small_alone or type == choice.small_multi: #small
            if COUNT:
                results = results.query('count == @bsm_count')
            if RSSI_BASE:
                results = results.query('max_rssi >= @bsm_max_rssi - 5').query('max_rssi <= @bsm_max_rssi - 2')
                results = results.query('ts_max_rssi >= @bsm_ts_max_rssi - @beacon_interval *1')\
                    .query('ts_max_rssi <= @bsm_ts_max_rssi + @beacon_interval*1') 
                results = results.query('min_rssi >= @bsm_min_rssi - 5').query('min_rssi <= @bsm_min_rssi - 2')
                results = results.query('ts_min_rssi >= @bsm_ts_min_rssi - @beacon_interval *1')\
                    .query('ts_min_rssi <= @bsm_ts_min_rssi + @beacon_interval*1')
                results = results.query('std_rssi >= @bsm_std_rssi - 0.5').query('std_rssi <= @bsm_std_rssi + 0.5 ')
                results = results.query('mean_rssi >= @bsm_mean_rssi -5').query('mean_rssi <= @bsm_mean_rssi - 2')
        else: #large
            if COUNT:
                results = results.query('count >= @bsm_count - 1').query('count <= @bsm_count + 1')
            if RSSI_BASE:
                results = results.query('max_rssi >= @bsm_max_rssi - 10').query('max_rssi <= @bsm_max_rssi - 2')
                results = results.query('ts_max_rssi >= @bsm_ts_max_rssi - @beacon_interval *2')\
                    .query('ts_max_rssi <= @bsm_ts_max_rssi + @beacon_interval*2')
                results = results.query('min_rssi >= @bsm_min_rssi - 10').query('min_rssi <= @bsm_min_rssi - 2')
                results = results.query('ts_min_rssi >= @bsm_ts_min_rssi - @beacon_interval *2')\
                    .query('ts_min_rssi <= @bsm_ts_min_rssi + @beacon_interval*2')               
                results = results.query('std_rssi >= @bsm_std_rssi - 1').query('std_rssi <= @bsm_std_rssi + 1 ')
                results = results.query('mean_rssi >= @bsm_mean_rssi -10').query('mean_rssi <= @bsm_mean_rssi - 2')
            
        merge = pd.concat([merge, results])  
        
        #end list
        
    if merge.empty is False:
        merge = merge.groupby(['SSID','BSSID']).agg(count=('BSSID', 'size')).reset_index()
        for bssid in attenance_tot.keys():
            cond = merge['BSSID'] == bssid
            merge.loc[cond, 'attendance'] = attenance_tot.get(bssid)
        merge.sort_values(by='count', ascending=False, inplace=True)
        merge = merge.query('attendance >= @length') 
        if merge.empty is False:
            max_count = merge.iloc[0]['count']
            possible_bssid_3 = merge.loc[merge['count'] == max_count]['BSSID']#only max count
        
    return possible_bssid_3


def match_pseudonym_wifi(dataframe, dataframe2, beacon_interval, sequences, match, type):
    """richiama le euristiche e salva l'associazione ottenuta"""
    
    to_remove_sequences = np.empty(0) #remove the lists after connected to wifi
    for list in tqdm(sequences):   
        
        first = True
        possible_bssid_tot = np.empty(0)
        if PEARSON:
            possible_bssid_1 = pearson(dataframe, dataframe2, beacon_interval, type, list) 
            first = False
            possible_bssid_tot = np.append(possible_bssid_tot, possible_bssid_1)               
            
        if TIME:
            possible_bssid_2 = time(dataframe, dataframe2, beacon_interval, type, list)
            if first:
                first = False
                possible_bssid_tot = np.append(possible_bssid_tot, possible_bssid_2)
            else:
                possible_bssid_tot = np.intersect1d(possible_bssid_tot, possible_bssid_2)
        
        if COUNT | RSSI_BASE:
            possible_bssid_3 = count_and_rssi(dataframe, dataframe2, beacon_interval, type, list)
            if first:
                first = False
                possible_bssid_tot = np.append(possible_bssid_tot, possible_bssid_3)
            else:
                possible_bssid_tot = np.intersect1d(possible_bssid_tot, possible_bssid_3)
        
        if len(possible_bssid_tot) == 0:    
            continue
   
        if type == choice.small_alone or type == choice.large_alone:
            if len(possible_bssid_tot) >1:
                continue
            else: 
                dataframe2, to_remove_sequences = insert(match, list, possible_bssid_tot[0], dataframe, dataframe2, beacon_interval, sequences, to_remove_sequences)
        else:
            dataframe2, to_remove_sequences = insert(match, list, possible_bssid_tot[0], dataframe, dataframe2, beacon_interval, sequences, to_remove_sequences)    
        
    #remove lists from sequences
    for index in sorted(to_remove_sequences.astype(int), reverse=True):
        del sequences[index]
    
    return match    


def filter_remaining_pseudonym(matched_pseudonyms_list,pseudonyms):
    """crea una lista di tutti gli pseudonimi non inseriti nella sequenza e ancora da controllare"""
    
    pseudonym_alone = []
    continuous_list = [element for sub_list in matched_pseudonyms_list for element in sub_list]

    for p in pseudonyms.tolist():
        if p not in continuous_list:
            pseudonym_alone.append([p])    
            
    return pseudonym_alone


def try_to_link(dataframe, events, old_pseudonym, new_pseudonym, results, pseudonyms, to_remove_pseudonyms):
    #in dataframe there are all the pseudonym
    #dallo pseudo risalgo al realid e li comparo
    old_realID = dataframe.loc[(dataframe['pseudonym'] == old_pseudonym)].iloc[0]['realID']   
    new_realID = dataframe.loc[(dataframe['pseudonym'] == new_pseudonym)].iloc[0]['realID']
    
    if old_realID == new_realID:
        results['tp'] += 1
    else:
        results['fp'] += 1

    #cancello per il conteggio dei falsi negativi
    events.drop(events[events['pseudonym'] == old_pseudonym].index, inplace=True)
    to_remove_pseudonyms = np.append(to_remove_pseudonyms, np.where(pseudonyms == old_pseudonym))
    
    return events, to_remove_pseudonyms         
            

def assign_pseudo_bssid(match, results_pseudo_bssid, dataframe, bssid_realID):
    """ Collega tra loro pseudonimi e BSSID e controlla le associazioni """
    
    for bssid in match.keys():
        realID_wifi = bssid_realID.loc[(bssid_realID['BSSID'] == bssid)].iloc[0]['realID']
        lists = match.get(bssid)
        for list in lists:
            for pseudonym in list:
                realID_ps = dataframe.loc[(dataframe['pseudonym'] == pseudonym) ].iloc[0]['realID']
                if realID_wifi == realID_ps:
                    results_pseudo_bssid['tp'] += 1
                else:
                    results_pseudo_bssid['fp'] += 1


def assign_changes_pseudo(dataframe, events, pseudonyms, results, match):
    """ Collega tra loro i nuovi pseudonimi trovati appartenenti allo stesso veicolo su zone separate e controlla le associazioni fatte """
    
    to_remove_pseudonyms = np.empty(0)
    for lists in match.values():
        length = len(lists)
        for i in range(0,length-1):
            events, to_remove_pseudonyms= try_to_link(dataframe, events, lists[i][-1], lists[i+1][0], results, pseudonyms, to_remove_pseudonyms)
    
    pseudonyms = np.delete(pseudonyms, to_remove_pseudonyms.astype(int))
    return pseudonyms    

def filter_dataframe(dataframe, pseudonyms):
    """Function which filter the data-frame by deleting the pseudonyms which occurs only once in the entire data-frame.
        (Delete the vehicles whit only one pseudonym?)
        
    Parameters
    ----------        
        data-frame : pandas.DataFrame, required
            The pandas dataframe which contains all the bsm eavesdropped by the antennas
                    
        pseudonyms : np.array, required
            The numpy array which contains the unique pseudonyms after the \'local_change\' function.

    Returns
    -------
        pseudonyms : np.array, required
            The updated numpy array which contains the unique pseudonyms. 

    """
    
    vehicles = np.array(dataframe['realID'].unique())
   
    to_remove_pseudonyms = np.empty(0)
    
    for v in vehicles.tolist():
        vehicles_pseudonyms = np.array(dataframe.loc[dataframe['realID'] == v]['pseudonym'].unique())
        #remove always the last pseudonym of a vehicle
        to_remove_pseudonyms = np.append(to_remove_pseudonyms, np.where(pseudonyms == vehicles_pseudonyms[-1]))
    
    pseudonyms = np.delete(pseudonyms, to_remove_pseudonyms.astype(int))
    return pseudonyms

def metrics_pseudo_bssid(results_pseudo_bssid, ps_tot):
    """Calculate and show the Precision, Recall and F1-Score metrics.
        associations pseudonyms/BSSID
    """
    
    tp = results_pseudo_bssid['tp']
    fp = results_pseudo_bssid['fp']
    fn = ps_tot-(tp+fp)
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))
    
    logging.info(f"{bcolors.GREEN}Join metrics: pseudonym/bssid{bcolors.RESET}")
    logging.info(f"{bcolors.GREEN}True positive {tp},  False positive {fp},  False negative {fn}{bcolors.RESET}")
    logging.info(f"{bcolors.GREEN}METRICS -> PRECISION: {'{:.5f}'.format(precision)}, RECALL: {'{:.5f}'.format(recall)}, F1 SCORE: {'{:.5f}'.format(f1_score)}{bcolors.RESET}\n")
    
    return precision, recall, f1_score


def local_results(results, fn):
    """Calculate and show the Precision, Recall and F1-Score metrics.

    Parameters
    ----------        
    results : dict, required
        A dictionary containing the key-value of the TP and the FP numbers.
        
    fn : integer, required
        The number representing the False Negative pseudonyms
    
    Returns
    -------
        precision : double
            The precision value

        recall : double
            The recall value 

        f1_score : double
            The F1-Score value 

    """
    tp = results['tp']
    fp = results['fp']
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))
    
    logging.info(f"{bcolors.BLUE}Join metrics: pseudonym/next pseudonym{bcolors.RESET}")
    logging.info(f"{bcolors.BLUE}True positive {tp},  False positive {fp},  False negative {fn}{bcolors.RESET}")
    logging.info(f"{bcolors.BLUE}METRICS -> PRECISION: {'{:.5f}'.format(precision)}, RECALL: {'{:.5f}'.format(recall)}, F1 SCORE: {'{:.5f}'.format(f1_score)}{bcolors.RESET}\n")
    
    return precision, recall, f1_score


def analyze(path, freq, dimensions):
    """This function sequentially call all the function of the python script.

    Parameters
    ----------        
        path : str, required
            The directory path where all the files are stored
                    
        freq : int, required
            The sending frequency of the bsm in the simulation.

        dimensions : bool, required
            Boolean value which if is True indicating the use of the vehicle size filter

    Returns
    -------
        precision : double
            The precision value

        recall : double
            The recall value 

        f1_score : double
            The F1-Score value

    """
    beacon_interval = 1/freq
    
    logging.info('Getting data from csv files...')
    dataframe, dataframe2, pseudonyms = getting_data(path)  #df concat of all rsu file
    
    pseudonyms_num = len(pseudonyms)
    bssid_realID = dataframe2[['BSSID', 'realID']]
    
    logging.info('Getting pseudonym change events...')
    #dataframe has all the lines
    events = pseudonym_change_events(dataframe.copy(), pseudonyms)

    logging.info('Checking for local pseudonym change...')
    results = {'tp': 0, 'fp': 0}
    
    #events -> pseudo remained; dataframe -> all dataframe
    pseudonyms, sequences = local_change(events, pseudonyms, beacon_interval, results, dimensions)  
    
    #copy of the list
    matched_pseudonyms_list= sequences[:]
    
    logging.info('Trying to match pseudonyms to wifi messages...')
    
    #link pseudonyms list to bssid
    match = {} 
     
    while True:
        seq_old = len(sequences) #control if change
        
        type = choice.small_alone
        for i in range(0, 4):
            #fill match
            match = match_pseudonym_wifi(dataframe, dataframe2, beacon_interval, sequences, match, type)
            type += 1  
        
        seq_new = len(sequences)
        
        if seq_old == seq_new:
            break
    
    #filtering the already matched pseudonyms, returns a list of list that can be used in the method saw before
    pseudonym_alone=filter_remaining_pseudonym(matched_pseudonyms_list,pseudonyms)    
    
    #match the remaining individual pseudonyms, a lot empty because have already problems
    while True:
        seq_old = len(pseudonym_alone)
        
        type = choice.small_alone
        for i in range(0, 4):
            match = match_pseudonym_wifi(dataframe, dataframe2, beacon_interval, pseudonym_alone, match, type)  
            type += 1  
        
        seq_new = len(pseudonym_alone)
        
        if seq_old == seq_new:
            break
    
    results_pseudo_bssid = {'tp': 0, 'fp': 0}
    assign_pseudo_bssid(match,results_pseudo_bssid,dataframe,bssid_realID)
    
    pseudonyms = assign_changes_pseudo(dataframe, events, pseudonyms, results, match)
    pseudonyms = filter_dataframe(events, pseudonyms)#erase last pseudo of every vehicle 

    fn = len(pseudonyms)
    precision1, recall1, f1_score1 = local_results(results, fn)
    precision2, recall2, f1_score2 = metrics_pseudo_bssid(results_pseudo_bssid, pseudonyms_num)
    return precision1, recall1, f1_score1, precision2, recall2, f1_score2 


def main(base_folder, freq, policy, dimensions):
    """This function compose the complete path using the base_folder, freq and policy and check if the folder actually exist.

    """
    path = f'{base_folder}/fq_{freq}Hz/pc_{policy}'
    path_if_directory(path)
    logging.info(f'Analyze data in \'{path}\'')
    
    precision1, recall1, f1_score1, precision2, recall2, f1_score2 = analyze(path, freq, dimensions)

    results_pseudo_bssid_file = 'results_pseudo_bssid.csv'
    if os.stat(results_pseudo_bssid_file).st_size == 0: # size in byte of the specific file
        head = True
    else:
        head = False
    with open(results_pseudo_bssid_file, 'a') as f:
        if head:
            f.write('fq,pc,prec,recall,f1_score,dimensions,count,rssi_base,pearson,time\n')
        f.write(f'{freq}, {policy}, {precision2}, {recall2}, {f1_score2}, {dimensions},     {COUNT}, {PEARSON}, {RSSI_BASE}, {TIME}\n')


    results_file = 'results.csv'
    if os.stat(results_file).st_size == 0: # size in byte of the specific file
        head = True
    else:
        head = False
    with open(results_file, 'a') as f:
        if head:
            f.write('fq,pc,prec,recall,f1_score,dimensionscount,rssi_base,pearson,time\\n')
        f.write(f'{freq}, {policy}, {precision1}, {recall1}, {f1_score1}, {dimensions},     {COUNT}, {PEARSON}, {RSSI_BASE}, {TIME}\n')
        


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
    parser.add_argument("-dim", "--dimensions", help="Consider vehicles dimensions",action="store_true")
    parser.add_argument("-count", "--count", help="Use counter", required=True, type=int, choices=[i for i in range(0,2)])
    parser.add_argument("-pears", "--pearson", help="Use pearson", required=True, type=int, choices=[i for i in range(0,2)])
    parser.add_argument("-rb", "--rssi_base", help="Use rssi base", required=True, type=int, choices=[i for i in range(0,2)])
    parser.add_argument("-tm", "--time", help="Use time", required=True, type=int, choices=[i for i in range(0,2)])
    args = parser.parse_args()
    
    global COUNT, PEARSON, RSSI_BASE, TIME
    if args.count:
        COUNT = True
    else:
        COUNT = False
    if args.pearson:
        PEARSON = True
    else:
        PEARSON = False
    if args.time:
        TIME = True
    else:
        TIME = False
    if args.rssi_base:
        RSSI_BASE = True
    else:
        RSSI_BASE = False
    if not(args.count or args.pearson or args.rssi_base or args.time) :
        assert False, 'No heuristic chosen'

    main(args.directory, args.freq, args.policy, args.dimensions)
