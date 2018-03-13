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
import time


def simulation(run,DATA,
               ch_rate,FE_outrate,
               FIFO_depth,FIFO_out_depth,
               FE_ch_latency,
               TE, TGAIN, sensors, events):

    data_out = np.array([]).reshape(0,6)
    lostP,lostC = 0,0
    Param = TOFPET.parameters(
                        ch_rate    = ch_rate,
                        FE_outrate = FE_outrate,   # 2.6Gb/s - 80 bits ch
                        FIFO_depth  = FIFO_depth,
                        FIFO_out_depth = FIFO_out_depth,
                        FE_ch_latency = FE_ch_latency,    # Max Wilkinson Latency
                        TE = TE,
                        TGAIN = TGAIN,
                        sensors = sensors,
                        events = events
                        )

    timing = np.random.randint(0,int(1E9/Param.ch_rate),size=events)
    # All sensors are given the same timestamp in an events
    sensor_id = range(64)
    env = simpy.Environment()

    # System Instanciation and Wiring
    # P = [TOFPET.producer(env,
    #                      data = DATA[:,i],
    #                      timing  = timing,
    #                      param = Param,
    #                      sensor_id = sensors[i])
    #                                     for i in range(len(Param.sensors))]
    # C = [TOFPET.FE_channel( env,
    #                         param = Param,
    #                         sensor_id = sensor[i])
    #                                     for i in range(len(Param.sensors))]
    # L = TOFPET.FE_outlink(  env,
    #                         data_out,
    #                         Param,
    #                         asic_id = 0)
    #
    # for i in range(Param.sensors):
    #     P[i].out = C[i]
    #     C[i].out = L

    ASIC_0 = TOFPET.FE_asic( env         = env,
                            param       = Param,
                            data        = DATA,
                            n_ch        = 64,
                            timing      = timing,
                            sensor_id   = sensor_id,
                            asic_id     = 0)

    env.run()

    for i in range(64):
        lostP = lostP + ASIC_0.Producer[i].lost
        lostC = lostC + ASIC_0.Channels[i].lost


    output = {  'lostP':lostP,
                'lostC':lostC,
                'log':ASIC_0.Link.log,
                'data_out':ASIC_0.Link.out_stream
                }

    return output



if __name__ == '__main__':

    disk  = TOFPET.hdf_access(  "/home/viherbos/DAQ_DATA/NEUTRINOS/",
                                "p_SET1.h5")
    DATA,sensors,n_events = disk.read()
    print (" NUMBER OF EVENTS IN SIMULATION: %d" % n_events)

    runs = 32
    latency = np.array([]).reshape(0,1)

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    sim_info={    'DATA'          : DATA,
                  'ch_rate'       : 300E3,
                  'FE_outrate'    : (2.6E9/80)/2,
                  'FIFO_depth'    : 4,
                  'FIFO_out_depth': 64*4,
                  'FE_ch_latency' : 5120,
                  'TE' : 7,
                  'TGAIN' : 1,
                  'sensors' : sensors,
                  'events' : n_events}

    # 2.6Gb/s - 80 bits ch
    # Max Wilkinson Latency

    start_time = time.time()

    mapfunc = partial(simulation, **sim_info)
    pool_output = pool.map(mapfunc, (i for i in range(runs)))

    pool.close()
    pool.join()



    lostP = [pool_output[j]['lostP'] for j in range(runs)]
    lostC = [pool_output[j]['lostC'] for j in range(runs)]
    outlink_ch = [ len(pool_output[j]['data_out'][:,0]) for j in range(runs)]
    total_ch_event = np.array(outlink_ch).sum()

    print ("A total of %d events processed" % total_ch_event)

    for i in range(runs):
        data_frame_array = pool_output[i]['data_out'][:,:]

        in_time_array  = data_frame_array[:,4]
        out_time_array = data_frame_array[:,5]
        latency_aux  = out_time_array - in_time_array
        #latency = np.vstack([latency,latency_aux.reshape(-1,1)])
        latency = np.pad(latency,((len(latency_aux),0),(0,0)),
                                 mode='constant',
                                 constant_values=0)
        # print latency.shape
        latency[0:len(latency_aux),0] = latency_aux


    elapsed_time = time.time()-start_time
    print ("IT TOOK %d SECONDS TO DO THIS" % elapsed_time)


    fit = fit_library.gauss_fit()
    fig = plt.figure(figsize=(16,4))
    fit(lostP,'sqrt')
    fit.plot(axis = fig.add_subplot(241),
            title = "FE FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(lostC,'sqrt')
    fit.plot(axis = fig.add_subplot(242),
            title = "Data Link FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(outlink_ch,'sqrt')
    fit.plot(axis = fig.add_subplot(243),
            title = "Recovered Channel Data",
            xlabel = "Number of Channel Events",
            ylabel = "Hits",
            res = False)
    fit(latency,50)
    fit.plot(axis = fig.add_subplot(244),
            title = "Data Latency",
            xlabel = "Latency in nanoseconds",
            ylabel = "Hits",
            res = False)

    fig.add_subplot(248)
    x_data = fit.bin_centers
    y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
    plt.plot(x_data,y_data)
    plt.ylim((0.9,1.0))
    plt.show()




#     output = mapfunc(1,**sim_info)
#
# print ("\n --------------------------------- \n")
# print ("FE Input Lost Events %d / Recovered Events %d\n" % (output['lostP'],
#                                                             len(output['data_out'])))
# print ("------------------------------------ \n")
#
# print ("\n --------------------------------- \n")
# print ("FE Output Lost Events %d / Recovered Events %d\n" % (output['lostC'],
#                                                             len(output['data_out'])))
# print ("------------------------------------ \n")

# plt.plot(output['log'][:,1],output['log'][:,0])
# plt.show()
