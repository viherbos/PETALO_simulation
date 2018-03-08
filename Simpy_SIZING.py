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

    disk  = TOFPET.hdf_access("/home/viherbos/TEMP/","processed.h5")
    DATA,sensors,events = disk.read()

    runs = 250
    latency = np.array([]).reshape(0,1)

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    sim_info={    'DATA'          : DATA,
                  'ch_rate'       : 200E3,
                  'FE_outrate'    : (2.6E9/80)/2,
                  # 2.6Gb/s - 80 bits ch
                  'FIFO_depth'    : 4,
                  'FIFO_out_depth': 64*4,
                  'FE_ch_latency' : 5120,
                  # Max Wilkinson Latency
                  'TE' : 2,
                  'TGAIN' : 1,
                  'sensors' : 64,
                  'events' : events}

    mapfunc = partial(simulation, **sim_info)
    pool_output = pool.map(mapfunc, (i for i in range(runs)))

    pool.close()
    pool.join()

    lostP = [pool_output[j]['lostP'] for j in range(runs)]
    lostC = [pool_output[j]['lostC'] for j in range(runs)]
    outlink_ch = [ len(pool_output[j]['data_out'][:,0]) for j in range(runs)]

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
    fit.plot(axis = fig.add_subplot(141),
            title = "FE FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(lostC,'sqrt')
    fit.plot(axis = fig.add_subplot(142),
            title = "Data Link FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(outlink_ch,'sqrt')
    fit.plot(axis = fig.add_subplot(143),
            title = "Recovered Channel Data",
            xlabel = "Number of Channel Events",
            ylabel = "Hits",
            res = False)
    fit(latency,'sqrt')
    fit.plot(axis = fig.add_subplot(144),
            title = "Data Latency",
            xlabel = "Latency in nanoseconds",
            ylabel = "Hits",
            res = False)

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
