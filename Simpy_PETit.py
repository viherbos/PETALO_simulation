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
from SimLib import TOFPET
from SimLib import HF_files as HF
import time


# def FE_sim(asic_id, DATA, timing, Param):
#
#     lostP,lostC = 0,0
#


#
# ASIC = TOFPET.FE_asic(  env         = env,
#                         param       = Param,
#                         data        = DATA[:,asic_id*64:(asic_id+1)*64],
#                         n_ch        = 64,
#                         timing      = timing,
#                         sensors     = sensors[asic_id*64:(asic_id+1)*64],
#                         asic_id     = asic_id)
#     env.run()
#
#     for i in range(64):
#         lostP = lostP + ASIC.Producer[i].lost
#         lostC = lostC + ASIC.Channels[i].lost
#
#
#     output = {  'lostP':lostP,
#                 'lostC':lostC,
#                 'data_out':ASIC.Link.out_stream,
#                 #'log':ASIC[0].Link.log
#                 }
#
#     return output


def FE_gen(threads,sim_info):

    env = simpy.Environment()

    start_time = time.time()

    ASIC = [TOFPET.FE_asic(
            env = env,
            param       = sim_info['Param'],
            data        = sim_info['DATA'][:,asic_id*64:(asic_id+1)*64],
            n_ch        = 64,
            timing      = sim_info['timing'],
            sensors     = sim_info['Param'].sensors[asic_id*64:(asic_id+1)*64],
            asic_id     = asic_id)
                for asic_id in threads]

    env.run()

    gen_output = [ASIC[i]() for i in range(len(threads))]


    elapsed_time = time.time()-start_time
    print ("IT TOOK %d SECONDS TO DO THIS" % elapsed_time)

    return gen_output



def FE_sim(asics, sim_info):

    kargs = {'sim_info':sim_info}

    FE_map = partial(FE_gen, **kargs)

    # Multiprocess Work
    pool_size = mp.cpu_count()//2
    pool = mp.Pool(processes=pool_size)

    pool_output = pool.map(FE_map, [[i] for i in asics])

    pool.close()
    pool.join()
    # pool_output = FE_map([1])

    return pool_output



if __name__ == '__main__':

    latency = np.array([]).reshape(0,1)
    files = range(5)
    A = HF.hdf_compose( "/home/viherbos/DAQ_DATA/NEUTRINOS/","p_SET_",files,256)
    DATA,sensors,n_events = A.compose()
    print (" NUMBER OF EVENTS IN SIMULATION: %d" % n_events)

    Param = TOFPET.parameters(
                        ch_rate    = 300E3,
                        FE_outrate = (2.6E9/80)/2,   # 2.6Gb/s - 80 bits ch
                        FIFO_depth  = 4,
                        FIFO_out_depth = 64*4,
                        FE_ch_latency = 5120,    # Max Wilkinson Latency
                        TE = 7,
                        TGAIN = 1,
                        sensors = sensors,
                        events = n_events
                        )

    timing = np.random.randint(0,int(1E9/Param.ch_rate),size=n_events)
    # All sensors are given the same timestamp in an events

    sim_info = {'DATA' : DATA, 'timing':timing, 'Param' : Param }

    pool_output = FE_sim([0,1,2,3],sim_info)

    print len(pool_output)
    print pool_output[0][0]

    asics = [0,1,2,3]
    lostP = [pool_output[j][0]['lostP'] for j in asics]
    # Total FIFO Drops on Producer stage
    lostC = [pool_output[j][0]['lostC'] for j in asics]
    # Total FIFO Drops on Outlink stage
    outlink_ch = [ len(pool_output[j][0]['data_out'][:,0]) for j in asics]
    # Number of data frames (channels) recovered at the output of each asic

    print ("A total of %d events processed" % np.array(outlink_ch).sum())

    for i in asics:
        data_frame_array = pool_output[i][0]['data_out'][:,:]

        in_time_array  = data_frame_array[:,4]
        out_time_array = data_frame_array[:,5]
        latency_aux  = out_time_array - in_time_array
        #latency = np.vstack([latency,latency_aux.reshape(-1,1)])
        latency = np.pad(latency,((len(latency_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        latency[0:len(latency_aux),0] = latency_aux





    fit = fit_library.gauss_fit()
    fig = plt.figure(figsize=(16,4))
    fit(lostP,'sqrt')
    fit.plot(axis = fig.add_subplot(231),
            title = "FE FIFO drops",
            xlabel = "Lost Events / Run",
            ylabel = "Hits",
            res = False)
    fit(lostC,'sqrt')
    fit.plot(axis = fig.add_subplot(234),
            title = "Data Link FIFO drops",
            xlabel = "Lost Events / Run",
            ylabel = "Hits",
            res = False)
    fit(outlink_ch,'sqrt')
    fit.plot(axis = fig.add_subplot(232),
            title = "Recovered Channel Data",
            xlabel = "Ch_Events / Run",
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
    plt.show()
