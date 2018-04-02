import numpy as np
import scipy as sp
import HF_files as HF
import config_sim as CFG
import pet_graphics as PG
import os
import pandas as pd



if __name__ == '__main__':
    CG = CFG.SIM_DATA()
    SHOW = PG.DET_SHOW(CG.data)

    DAQ_data = HF.DAQ_IO("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
                            "daq_output.h5",
                            "p_FRSET_0.h5")
    data,sensors = DAQ_data.read()

    # DATA FRAME
    # aux_list = {'data'      :   nparray[0],
    #             'event'     :   nparray[1],
    #             'sensor_id' :   nparray[2],
    #             'asic_id'   :   nparray[3],
    #             'in_time'   :   nparray[4],
    #             'out_time'  :   nparray[5]}

    order = np.argsort(data[:,4])
    data_ordered = data[order,:]

    print data_ordered[:10,:]

    # Event Extraction
    timestamp = data_ordered[0,4]
    index = np.argwhere(data_ordered[:,4]==timestamp)


    event = data_ordered[index[:,0]]


    # Sensor positions
    os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    filename = "p_FRSET_0.h5"
    positions = np.array(pd.read_hdf(filename,key='sensors'))

    active_positions=np.array([]).reshape(0,4)
    for i in event[:,2]:
        active_positions = np.pad(active_positions,((0,1),(0,0)),
                                                    mode='constant',
                                                    constant_values=0)
        index = np.argwhere(positions[:,0]==i)
        active = positions[index[:,0]]
        active_positions[-1,:] = active[0,:]

    print active_positions

    event_data = event[:,0].reshape(1,len(event[:,0]))
    print event_data.shape
    SHOW(active_positions,event_data,0,False,True)
