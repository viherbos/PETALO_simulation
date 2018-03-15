import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
import os
import pandas as pd



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
            files   :   array of files
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

        for i in range(1,len(self.files)):
            hf = hdf_access(self.path,self.file_name + str(self.files[i]) + ".h5")
            self.data_aux,self.fake,self.events = hf.read()
            self.data = np.pad( self.data,
                                ((self.events,0),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.data[0:self.events,:] = self.data_aux


        return self.data, self.sensors, self.data.shape[0]


def main():

    TEST_c = hdf_compose(  "/home/viherbos/DAQ_DATA/NEUTRINOS/",
                           "p_SET_",[0,1,2,3],256)
    a,b,c = TEST_c.compose()

    print a
    print b
    print c


if __name__ == "__main__":
    main()
