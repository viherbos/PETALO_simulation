import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
import os
import pandas as pd
import time


class daq_out(object):

    def __init__(self,path,out_filename,ref_filename,data):
        self.path = path
        self.out_filename = out_filename
        self.ref_filename = ref_filename
        self.data = data

    def write(self):
        os.chdir(self.path)
        with pd.HDFStore(self.out_filename) as store:
            panel_array = pd.DataFrame( data = self.data,
                                        columns = ['data','event','sensor_id',
                                                    'asic_id','in_time','out_time'])
            sensors_xyz = np.array( pd.read_hdf(self.ref_filename,
                                        key='sensors'),
                                        dtype = 'float32')
            sensors_array = pd.DataFrame( data=sensors_xyz,
                                          columns=['sensor','x','y','z'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',panel_array)
            store.put('sensors',sensors_array)
            store.close()

class hdf_access(object):
    """ A utility class to access data in hf5 format.
        read method is used to load data from a preprocessed file.
        The file format is a table with each column is a sensor and
        each row an event
    """

    def __init__(self,path,file_name):
        self.path = path
        self.file_name = file_name

    def read(self):
        os.chdir(self.path)
        self.data = pd.read_hdf(self.file_name,key='MC')

        # Reads translated hf files (table with sensor/charge per event)
        self.sensors = np.array(self.data.columns)
        self.data = np.array(self.data, dtype = 'int32')
        self.events = self.data.shape[0]

        #returns data array, sensors vector, and number of events
        return self.data,self.sensors,self.events


class hdf_compose(object):
    """ A utility class to access preprocessed data from MCs in hf5 format.
            param
            files           : Array of files
            n_sensors       : Number of sensors (all of them)
            Output
            composed data
            sensor array
            number of events
    """

    def __init__(self,path,file_name,files,n_sensors):
        self.path       = path
        self.file_name  = file_name
        self.files      = files
        self.n_sensors  = n_sensors
        self.data       = np.array([]).reshape(0,self.n_sensors)
        self.data_aux   = np.array([]).reshape(0,self.n_sensors)

    def compose(self):

        hf = hdf_access(self.path,self.file_name + str(self.files[0]) + ".h5")
        self.data_aux,self.sensors,self.events = hf.read()
        self.data = np.pad( self.data,
                            ((self.events,0),(0,0)),
                            mode='constant',
                            constant_values=0)
        self.data[0:self.events,:] = self.data_aux

        for i in self.files:
            hf = hdf_access(self.path,self.file_name + str(i) + ".h5")
            self.data_aux,self.fake,self.events = hf.read()
            self.data = np.pad( self.data,
                                ((self.events,0),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.data[0:self.events,:] = self.data_aux


        return self.data, self.sensors, self.data.shape[0]


def main():

    start = time.time()

    files = [0,1,2,3,4,5,6,8]

    TEST_c = hdf_compose(  "/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
                           "p_FRSET_",files,1536)
    a,b,c = TEST_c.compose()

    time_elapsed = time.time() - start

    print ("It took %d seconds to compose %d files" % (time_elapsed,
                                                       len(files)))


if __name__ == "__main__":
    main()
