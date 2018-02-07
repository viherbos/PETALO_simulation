import os
import pandas as pd
import numpy as np


class HF(object):
    def __init__(self,path,in_file,out_file):
        self.path     = path
        self.in_file  = in_file
        self.out_file = out_file
        self.waves    = np.array([])
        self.extents  = np.array([])
        self.sensors  = np.array([])
        self.n_events = 0
        self.out_table = np.array([])


    def read(self):
        os.chdir(self.path)
        self.waves   = np.array( pd.read_hdf(self.in_file,key='Run/waveforms'),
                            dtype = 'int32')
        self.extents = np.array( pd.read_hdf(self.in_file,key='Run/extents'),
                            dtype = 'int32')
        self.n_events = self.extents.shape[0]

        sensors_t = np.array( pd.read_hdf(self.in_file,key='Run/sensor_positions'),
                            dtype = 'int32')
        self.sensors = sensors_t[:,0]

    def write(self):
        with pd.HDFStore(self.out_file,
                        complevel=9,
                        complib='bzip2') as store:
            panel_array = pd.DataFrame( data=self.out_table,
                                        columns=self.sensors)
            store.put('run',panel_array)
            store.close()


    def process(self):
        self.out_table = np.zeros((self.n_events,self.sensors.shape[0]),dtype='int32')
        low_limit = 0

        for i in range(0,self.n_events-1):
            high_limit = self.extents[i,1]
            event_wave = self.waves[low_limit:high_limit+1,:]

            for j in range(0,self.sensors.shape[0]):
                condition   = (event_wave[:,0]==self.sensors[j])
                sensor_data = np.sum(event_wave[condition,2])
                self.out_table[i,j] = sensor_data


def main():
    TEST_c = HF(  "/home/viherbos/TEMP/",
                  "petit_prova.pet.h5",
                  "processed.h5")
    TEST_c.read()
    TEST_c.process()
    TEST_c.write()



if __name__ == "__main__":
    main()
