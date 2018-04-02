import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_analysis/")
import fit_library
import HF_translator as HFT
import os
import multiprocessing as mp
from functools import partial
from SimLib import DAQ_RING
from SimLib import HF_files as HF
import time
from SimLib import config_sim as CFG
from SimLib import pet_graphics as PG
import pandas as pd


def DAQ_gen(threads,sim_info):

    env = simpy.Environment()

    start_time = time.time()

    block_size = sim_info['Param'].FE_n_channels*sim_info['Param'].n_asics

    L1h = [DAQ_RING.L1_hierarchy(
            env = env,
            param       = sim_info['Param'],
            DATA        = sim_info['DATA'][:,i*block_size:(i+1)*block_size],
            timing      = sim_info['timing'],
            sensors     = sim_info['Param'].sensors[i*block_size:(i+1)*block_size],
            L1_id       = i)
                for i in threads]

    env.run()

    gen_output = [L1h[i]() for i in range(len(threads))]


    elapsed_time = time.time()-start_time
    print ("IT TOOK %d SECONDS TO DO THIS" % elapsed_time)

    return gen_output



def DAQ_sim(L1h_array, sim_info):

    kargs = {'sim_info':sim_info}
    DAQ_map = partial(DAQ_gen, **kargs)

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    pool_output = pool.map(DAQ_map, [[i] for i in L1h_array])

    pool.close()
    pool.join()
    #pool_output = DAQ_map([1])

    return pool_output



if __name__ == '__main__':

    CG = CFG.SIM_DATA()
    # Read data from json file
    n_sipms = CG.data['TOPOLOGY']['sipm_int_row']*CG.data['TOPOLOGY']['n_rows'] + \
              CG.data['TOPOLOGY']['sipm_ext_row']*CG.data['TOPOLOGY']['n_rows']
    # Work out number of SiPMs based on geometry data
    n_asics = n_sipms // CG.data['TOFPET']['n_channels']
    print ("Number of SiPM : %d \n Number of ASICS : %d" % (n_sipms,n_asics))

    n_files = 1
    #Number of files to group for data input
    A = HF.hdf_compose( "/home/viherbos/DAQ_DATA/NEUTRINOS/RING/","p_FRSET_",
                        range(n_files),n_sipms)
    DATA,sensors,n_events = A.compose()

    n_events = 1000
    DATA = DATA[0:1000,:]

    SHOW = PG.DET_SHOW(CG.data)
    os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    filename = "p_FRSET_0.h5"
    positions = np.array(pd.read_hdf(filename,key='sensors'))
    data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    SHOW(positions,data,0,True,False)

    print (" %d EVENTS IN %d H5 FILES" % (n_events,n_files))

    Param = DAQ_RING.parameters(
                    ch_rate    = CG.data['ENVIRONMENT']['ch_rate'],
                    FE_outrate = CG.data['TOFPET']['outlink_rate'],
                    FIFO_depth  = CG.data['TOFPET']['IN_FIFO_depth'],
                    FIFO_out_depth = CG.data['TOFPET']['OUT_FIFO_depth'],
                    FE_ch_latency = CG.data['TOFPET']['MAX_WILKINSON_LATENCY'],
                    TE = CG.data['TOFPET']['TE'],
                    TGAIN = CG.data['TOFPET']['TGAIN'],
                    FE_n_channels = CG.data['TOFPET']['n_channels'],
                    sensors = sensors,
                    events = n_events,
                    L1_outrate = CG.data['L1']['L1_outrate'],
                    FIFO_L1_depth = CG.data['L1']['FIFO_L1_depth'],
                    n_asics = CG.data['L1']['n_asics']
                    )

    timing = np.random.randint(0,int(1E9/Param.ch_rate),size=n_events)
    # All sensors are given the same timestamp in an events

    sim_info = {'DATA' : DATA, 'timing':timing, 'Param' : Param }

    L1s_n = CG.data['L1']['n_L1']
    L1s = range(L1s_n)
    pool_output = DAQ_sim(L1s,sim_info)

    outlink_ch = [ len(pool_output[j][0]['data_out'][:,0]) for j in L1s]
    # Number of data frames (channels) recovered at the output of each L1

    print ("A total of %d events processed" % np.array(outlink_ch).sum())

    latency = np.array([]).reshape(0,1)
    lostP = np.array([]).reshape(0,1)
    lostC = np.array([]).reshape(0,1)
    lostL1 = np.array([]).reshape(0,1)
    data   = np.array([]).reshape(0,6)


    for i in L1s:
        data_frame_array = pool_output[i][0]['data_out'][:,:]
        lostP_aux  = pool_output[i][0]['lostP']
        lostC_aux  = pool_output[i][0]['lostC']
        lostL1_aux = pool_output[i][0]['lostL1']


        in_time_array  = data_frame_array[:,4]
        out_time_array = data_frame_array[:,5]
        latency_aux  = out_time_array - in_time_array
        #latency = np.vstack([latency,latency_aux.reshape(-1,1)])
        latency = np.pad(latency,((len(latency_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        latency[0:len(latency_aux),0] = latency_aux

        lostP = np.pad(lostP,((len(lostP_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        lostP[0:len(lostP_aux),0] = lostP_aux[:,0]

        lostC = np.pad(lostC,((len(lostC_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        lostC[0:len(lostC_aux),0] = lostC_aux[:,0]

        lostL1 = np.pad(lostL1,((len(lostL1_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        lostL1[0:len(lostL1_aux),0] = lostL1_aux[:,0]

        data = np.pad(data,((data_frame_array.shape[0],0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        data[0:data_frame_array.shape[0],:] = data_frame_array


    fit = fit_library.gauss_fit()
    fig = plt.figure(figsize=(16,8))
    fit(lostP,'sqrt')
    fit.plot(axis = fig.add_subplot(231),
            title = "FE FIFO drops",
            xlabel = "Lost Events",
            ylabel = "Hits",
            res = False)
    fit(lostC,'sqrt')
    fit.plot(axis = fig.add_subplot(234),
            title = "Data Link FIFO drops",
            xlabel = "Lost Events",
            ylabel = "Hits",
            res = False)
    fit(lostL1,'sqrt')
    fit.plot(axis = fig.add_subplot(235),
            title = "L1 FIFO drops",
            xlabel = "Lost Events",
            ylabel = "Hits",
            res = False)
    fit(outlink_ch,'sqrt')
    fit.plot(axis = fig.add_subplot(232),
            title = "Recovered Channel Data",
            xlabel = "Ch_Events",
            ylabel = "Hits",
            res = False)
    fit(latency,50)
    fit.plot(axis = fig.add_subplot(233),
            title = "Data Latency",
            xlabel = "Latency in nanoseconds",
            ylabel = "Hits",
            res = False)


    fig.add_subplot(236)
    x_data = fit.bin_centers
    y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
    plt.plot(x_data,y_data)
    plt.ylim((0.9,1.0))

    fig.tight_layout()

    plt.show()


    DAQ_dump = HF.DAQ_IO("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
                            "daq_output.h5",
                            "p_FRSET_0.h5")
    DAQ_dump.write(data)
