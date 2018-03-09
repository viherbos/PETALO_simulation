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


def simulation(run,DATA,
               ch_rate,FE_outrate,
               FIFO_depth,FIFO_out_depth,
               FE_ch_latency,
               TE, TGAIN, sensors, events):

    data_out = np.array([]).reshape(0,3)
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
    env = simpy.Environment()

    # System Instanciation and Wiring
    P = [TOFPET.producer(env,
                        data = DATA[:,i],
                        timing  = timing,
                        param = Param) for i in range(Param.sensors)]
    C = [TOFPET.FE_channel(env,
                        param=Param) for i in range(Param.sensors)]
    L = TOFPET.FE_outlink(env,data_out,Param)

    for i in range(Param.sensors):
        P[i].out = C[i]
        C[i].out = L


    env.run()

    for i in range(Param.sensors):
        lostP = lostP + P[i].lost
        lostC = lostC + C[i].lost

    output = {  'lostP':lostP,
                'lostC':lostC,
                'log':L.log,
                'data_out':L.out_stream
                }

    return output



if __name__ == '__main__':

    disk  = TOFPET.hdf_access("./","processed.h5")
    DATA,sensors,events = disk.read()

    runs = 250
    latency = np.array([]).reshape(0,1)

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    sim_info={    'DATA'          : DATA,
                  'ch_rate'       : 200E3,
                  'FE_outrate'    : (2.6E9/80),
                  'FIFO_depth'    : 4,
                  'FIFO_out_depth': 64*4,
                  'FE_ch_latency' : 5120,
                  'TE' : 8,
                  'TGAIN' : 1,
                  'sensors' : 64,
                  'events' : events}

    # 2.6Gb/s - 80 bits ch
    # Max Wilkinson Latency

    mapfunc = partial(simulation, **sim_info)
    pool_output = pool.map(mapfunc, (i for i in range(runs)))

    pool.close()
    pool.join()

    lostP = [pool_output[j]['lostP'] for j in range(runs)]
    lostC = [pool_output[j]['lostC'] for j in range(runs)]
    outlink_ch = [ len(pool_output[j]['data_out'][:,0]) for j in range(runs)]
    total_ch_event = np.array(outlink_ch).sum()
    print total_ch_event

    for i in range(runs):
        data_aux = np.array(pool_output[i]['data_out'])
        in_time  = data_aux[:,1]
        out_time = data_aux[:,2]
        latency_aux  = out_time - in_time
        latency = np.vstack([latency,latency_aux.reshape(-1,1)])
        data_out = data_aux[:,0]


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
