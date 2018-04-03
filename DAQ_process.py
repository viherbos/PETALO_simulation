import numpy as np
import scipy as sp
from SimLib import HF_files as HF
import HF_translator as HFT
from SimLib import config_sim as CFG
from SimLib import pet_graphics as PG
import os
import pandas as pd



if __name__ == '__main__':
    CG = CFG.SIM_DATA()
    SHOW = PG.DET_SHOW(CG.data)

    DAQ_data = HF.DAQ_IO("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
                            "daq_output.h5",
                            "p_FRSET_0.h5",
                            "daq_output_processed.h5")
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

    # Event Extraction only active sensors
    event_pointer = 0
    events = 5000
    QDC_table = np.zeros((events,sensors.shape[0]),dtype='int32')


    for j in range(events):
        timestamp = data_ordered[event_pointer,4]
        index = np.argwhere(data_ordered[:,4]==timestamp)
        event = data_ordered[index[:,0]]
        event_pointer += event.shape[0]

        positions = sensors

        active_positions=np.array([]).reshape(0,4)

        pos = 0
        for i in event[:,2]:
            active_positions = np.pad(active_positions,((0,1),(0,0)),
                                                        mode='constant',
                                                        constant_values=0)
            index = np.argwhere(positions[:,0]==i)
            active = positions[index[:,0]]
            active_positions[-1,:] = active[0,:]

            QDC_table[j,index[:,0]] =  event[pos,0]
            pos+=1

        event_data = event[:,0].reshape(1,len(event[:,0]))

        #SHOW(active_positions,event_data,0,False,True)

    DAQ_data.write_out(QDC_table)

    for i in range(20):
        SHOW(sensors,QDC_table,i,False,True)
