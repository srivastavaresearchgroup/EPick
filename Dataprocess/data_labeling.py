import os
import sys
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy import read, Stream
from data_pipeline import DataWriter

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream
    '''
    data = np.array(dataset)
              
    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']
    
    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']
    
    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])
    return stream

file_name = "./.hdf5"
catalog_list  = "./.csv"

# for seismic evevt reading the csv file into a dataframe:
df = pd.read_csv(catalog_list)
df = df[(df.trace_category == 'earthquake_local')]
df = df.drop_duplicates(subset=['source_id'], keep='first')

dtfl = h5py.File(file_name, 'r')

## example for STEAD dataset 
for c, evi in enumerate(df['source_id'].to_list()[:100]):
    # evi denotes the trace name
    trace = df[(df.source_id)== str(evi)]['trace_name']
    trace = list(str(trace).split(' ')[4].split('\n'))[0]
    dataset = dtfl.get('data/'+str(trace)) 

    p_st = df[(df.trace_name)==str(trace)]['p_arrival_sample']
    p_st = list(str(p_st).split(' ')[4].split('\n'))[0]
    p_st = float(p_st)

    s_st = df[(df.trace_name)==str(trace)]['s_arrival_sample']
    s_st = list(str(s_st).split(' ')[4].split('\n'))[0]
    s_st = float(s_st)

    #waveforms, 3 channels: first row: Z channel, second row: N channel, third row:E channel 
    st = make_stream(dataset)
    # window_size = 60
    st_event =st
    
    ## label initialization
    label_obj = st_event.copy()
    label_obj[0].data[...] = 1
    label_obj[1].data[...] = 0
    label_obj[2].data[...] = 0
    
    ## labeling arrival time of P and S waves
    label_obj[1].data[int(p_st)] = 1
    label_obj[2].data[int(s_st)] = 2

    traces = Stream()
    label_obj[0].stats.channel="N"
    label_obj[1].stats.channel="P"
    label_obj[2].stats.channel="S"
    traces += label_obj
    traces.normalize().plot()
    traces = np.array(label_obj)

    '''tfrecords file'''
    output_name = "./"+ str(evi)+'.tfrecords'
    output_path = os.path.join('.', output_name)
    writer = DataWriter(output_path)
    writer.write(st_event.copy().normalize(), label_obj)
