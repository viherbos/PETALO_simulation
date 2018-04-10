import os
import pandas as pd
import tables as tb
import numpy as np
import multiprocessing as mp
from functools import partial


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

            # Ask PAOLA about the channel grouping (SiPM -> ASIC)+++++
            # or change ASIC grouping here (ARGHH!!!)

            low_limit = high_limit+1
            #print ("EVENT %d processed" % i)
            count_a = 0

        # Rows -> Events // Columns -> Sensors (Ordered by detector, 64 each ASIC)

class HF_CONT(object):
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
        self.gamma1_i1 = np.array([])
        self.gamma2_i1 = np.array([])
        self.table = None


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
        self.sensors_order = np.argsort(self.sensors)
        self.sensors = self.sensors[self.sensors_order]

    def read_table(self):
        os.chdir(self.path)
        h5file = tb.open_file(self.in_file, mode="r")
        self.table = h5file.root.MC.particles
        #particle_indx = [table['particle_indx'] for x in table.iterrows()]


    def write(self,iter=False):
        with pd.HDFStore(self.out_file) as store:
            self.panel_array = pd.DataFrame( data=self.out_table,
                                             columns=self.sensors)
            self.sensors_xyz = np.array( pd.read_hdf(self.in_file,
                                        key='MC/sensor_positions'),
                                        dtype = 'float32')
            self.sensors_order = np.argsort(self.sensors_xyz[:,0])
            self.sensors_array = pd.DataFrame( data=self.sensors_xyz[self.sensors_order,:],
                                                columns=['sensor','x','y','z'])

            if (iter == True):
                # Do we need first iter of gammas?

                self.iter1_array = pd.DataFrame( data=np.concatenate((self.gamma1_i1,
                                                                      self.gamma2_i1),
                                                                      axis=1),
                                                 columns=['x1','y1','z1','x2','y2','z2'])
                store.put('iter1',self.iter1_array)

            store.put('MC',self.panel_array)
            store.put('sensors',self.sensors_array)
            store.close()

        # complevel and complib are not compatible with MATLAB
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


        for i in range(0,self.n_events):
            high_limit = self.extents[i,1]
            event_wave = self.waves[low_limit:high_limit+1,:]

            for j in self.sensors:
                condition   = (event_wave[:,0] == j)
                sensor_data = np.sum(event_wave[condition,2])
                self.out_table[i,count_a] = sensor_data
                #self.sensors[count_a] = count
                count_a += 1

            low_limit = high_limit+1
            count_a = 0


    def process_table(self):
        self.gamma1_i1 = np.zeros((self.n_events,3),dtype='float32')
        self.gamma2_i1 = np.zeros((self.n_events,3),dtype='float32')
        low_limit = 0
        count = 0
        count_a = 0

        for i in range(0,self.n_events):
            high_limit = self.extents[i,4]
            event_particles = self.table[low_limit:high_limit+1]

            cond1 = np.array(event_particles[:]['particle_name']=="e-")
            cond2 = np.array(event_particles[:]['initial_volume']=="ACTIVE")
            cond3 = np.array(event_particles[:]['final_volume']=="ACTIVE")
            cond4 = np.array(event_particles[:]['mother_indx']==1)
            cond5 = np.array(event_particles[:]['mother_indx']==2)


            A1 = event_particles[cond1 * cond2 * cond3 * cond4]
            A2 = event_particles[cond1 * cond2 * cond3 * cond5]

            if len(A1)==0:
                self.gamma1_i1[i,:] = np.zeros((1,3))
            else:
                A1_index = A1[:]['initial_vertex'][:,3].argmin()
                self.gamma1_i1[i,:] = A1[A1_index]['initial_vertex'][0:3]

            if len(A2)==0:
                self.gamma2_i1[i,:] = np.zeros((1,3))
            else:
                A2_index = A2[:]['initial_vertex'][:,3].argmin()
                self.gamma2_i1[i,:] = A2[A2_index]['initial_vertex'][0:3]


            low_limit = high_limit+1


def TRANS_gen(index,path,filename,outfile):

    TEST_c = HF_CONT( path,filename + str(index) +".pet.h5",
                      outfile + str(index) + ".h5" )
    TEST_c.read()
    TEST_c.process()
    TEST_c.read_table()
    TEST_c.process_table()
    TEST_c.write(iter=True)




if __name__ == "__main__":

    kargs = {'path'     :"/home/viherbos/DAQ_DATA/NEUTRINOS/Small_Animal/",
             'filename' :"full_ring_infinity_depth3cm.",
             'outfile'  :"p_FR_infinity_"}
    TRANS_map = partial(TRANS_gen, **kargs)

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    pool.map(TRANS_map, [i for i in range(2)])

    pool.close()
    pool.join()



# for i in range(1):
#     TEST_c = HF(  "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit",
#                   "LXe_SiPM9mm2_xyz5cm_"+str(i)+".pet.h5",
#                   "p_SET_" + str(i) + ".h5" )
#     TEST_c.read()
#     TEST_c.process()
#     TEST_c.write()
#
#     print ("%d files processed" % i)

# for i in [0,1,2,3,4,5,6,8]:
#     TEST_c = HF(  "/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
#                   "full_ring_SiPM9mm2."+str(i)+".pet.h5",
#                   "p_FRSET_" + str(i) + ".h5" )
#     TEST_c.read()
#     TEST_c.process()
#     TEST_c.write()
#
#     print ("%d files processed" % i)

# for i in [0,1]:
#     TEST_c = HF_CONT(  "/home/viherbos/DAQ_DATA/NEUTRINOS/Small_Animal/",
#                   "full_ring_infinity_depth3cm."+str(i)+".pet.h5",
#                   "p_FR_infinity_" + str(i) + ".h5" )
#     TEST_c.read()
#     TEST_c.process()
#     TEST_c.write()
#
#     print ("%d files processed" % i)
