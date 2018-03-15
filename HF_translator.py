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
        self.sensors_t = np.array([])


    def read(self):
        os.chdir(self.path)
        self.waves   = np.array( pd.read_hdf(self.in_file,key='MC/waveforms'),
                            dtype = 'int32')
        self.extents = np.array( pd.read_hdf(self.in_file,key='MC/extents'),
                            dtype = 'int32')
        self.n_events = self.extents.shape[0]

        self.sensors_t = np.array( pd.read_hdf(self.in_file,key='MC/sensor_positions'),
                            dtype = 'int32')
        self.sensors = self.sensors_t[:,0]

    def write(self):
        with pd.HDFStore(self.out_file) as store:
            self.panel_array = pd.DataFrame( data=self.out_table,
                                             columns=self.sensors)
            self.sensors_xyz = np.array( pd.read_hdf(self.in_file,
                                        key='MC/sensor_positions'),
                                        dtype = 'float32')
            self.sensors_order = np.argsort(self.sensors_xyz[:,0])
            self.sensors_array = pd.DataFrame( data=self.sensors_xyz[self.sensors_order,:],
                                                columns=['sensor','x','y','z'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',self.panel_array)
            store.put('sensors',self.sensors_array)
            store.close()
        # panel_array = pd.DataFrame( data=self.out_table,columns=self.sensors)
        # panel_array.to_hdf( self.out_file,
        #                     key = 'charge',
        #                     format ='fixed',
        #                     complevel = 9,
        #                     complib = 'bzip2')


    def process(self):
        self.out_table = np.zeros((self.n_events,self.sensors.shape[0]),dtype='int32')
        low_limit = 0
        count = 0
        count_a = 0
        self.ch_asic     = 64
        self.n_detectors = int(self.sensors.max()/1000)
        self.asics_detec = np.sum((self.sensors>=1000)*(self.sensors<2000))/self.ch_asic


        for i in range(0,self.n_events):
            high_limit = self.extents[i,1]
            event_wave = self.waves[low_limit:high_limit+1,:]

            for j in range(0,self.n_detectors):
                count = 1000*(j+1)
                for l in range(0,self.asics_detec):
                    for m in range(64):
                        #print ("cout %d // count_a %d" % (count,count_a))
                        condition   = (event_wave[:,0]==count)
                        sensor_data = np.sum(event_wave[condition,2])
                        self.out_table[i,count_a] = sensor_data
                        self.sensors[count_a] = count
                        count += 1
                        count_a += 1

            low_limit = high_limit+1
            #print ("EVENT %d processed" % i)
            count_a = 0

        # Rows -> Events // Columns -> Sensors (Ordered by detector, 64 each ASIC)

def main():

    for i in range(1):
        TEST_c = HF(  "/home/viherbos/DAQ_DATA/NEUTRINOS/",
                      "LXe_SiPM9mm2_xyz5cm_"+str(i)+".pet.h5",
                      "p_SET_" + str(i) + ".h5" )
        TEST_c.read()
        TEST_c.process()
        TEST_c.write()

        print ("%d files processed" % i)


if __name__ == "__main__":
    main()
