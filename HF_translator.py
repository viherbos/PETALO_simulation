import os
import pandas as pd
import numpy as np


class HF_inspector(object):
    def __init__(self,path,in_file,out_file):
        self.path     = path
        self.in_file  = in_file
        self.out_file = out_file
        self.waves    = np.array([])
        self.extents  = np.array([])
        self.sensors  = np.array([])
        self.n_events = 0


    def HF_read(self):
        os.chdir(self.path)
        self.waves   = np.array( pd.read_hdf(self.in_file,key='Run/waveforms'),
                            dtype = 'int32')
        self.extents = np.array( pd.read_hdf(self.in_file,key='Run/extents'),
                            dtype = 'int32')
        self.n_events = extents.shape[0]

        sensors_t = np.array( pd.read_hdf(self.in_file,key='Run/sensor_positions'),
                            dtype = 'int32')
        self.sensors = sensors_t[:,0]




    def HF_sensor




def main():
    TEST_c = HF_inspector(  "/home/viherbos/TEMP/",
                            "petit_prova.pet.h5",
                            "processed.h5")



if __name__ == "__main__":
    main()
