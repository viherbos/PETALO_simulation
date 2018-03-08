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
import pandas as pd
import multiprocessing as mp
from functools import partial


class Full(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class parameters(object):

    def __init__(self,ch_rate,FE_outrate,
                 FIFO_depth,FIFO_out_depth,
                 FE_ch_latency,TE,TGAIN,
                 sensors,events):
        self.ch_rate     = ch_rate
        self.FE_outrate  = FE_outrate
        self.FIFO_depth  = FIFO_depth
        self.sensors     = sensors
        self.events      = events
        self.FIFO_out_depth = FIFO_out_depth
        self.FE_ch_latency  = FE_ch_latency      # Maximum Ch Latency
        self.TE        = TE
        self.TGAIN      = TGAIN


class hdf_access(object):

    def __init__(self,path,file_name):
        self.path = path
        self.file_name = file_name

    def read(self):
        os.chdir(self.path)
        self.data = np.array( pd.read_hdf(self.file_name,key='run'),
                                          dtype = 'int32')
        # Reads translated hf files (table with sensor/charge per event)
        self.events = self.data.shape[0]
        self.sensors = self.data.shape[1]

        return self.data,self.sensors,self.events



class producer(object):

    def __init__(self,env,data,timing,param):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())
        self.counter = 0
        self.lost = 0
        self.data = data
        self.timing = timing
        self.TE = param.TE

    def run(self):
        while self.counter < len(self.data):

            yield self.env.timeout(int(self.timing[self.counter]))
            #print_stats(env,self.out.res)

            try:
                if self.data[self.counter]>self.TE:
                    self.lost = self.out.put(self.data[self.counter],self.lost)
                self.counter += 1
                # Drop data. FIFO is FULL so data is lost
            except IndexError:
                print "List Empty"


class FE_channel(object):

    def __init__(self,env,param):
        self.env = env
        self.FIFO_size = param.FIFO_depth
        self.res = simpy.Store(self.env,capacity=self.FIFO_size)
        self.action = env.process(self.run())
        self.out = None
        self.latency = param.FE_ch_latency
        self.index = 0
        self.lost = 0
        self.gain = param.TGAIN
        self.log = []

    def print_stats(self):
        #print ('TIME: %d // ITEMS: %s ' % (self.env.now,self.res.items))
        self.log.append((len(self.res.items),self.env.now))

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_size):
                self.res.put(data)
                #self.print_stats()
                return lost
            else:
                raise Full('FIFO is FULL')
        except Full as e:
            #print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            self.msg = yield self.res.get()
            self.wilk_delay = int((self.latency/1024)*self.msg*self.gain)
            if self.wilk_delay > self.latency:
                self.wilk_delay = self.latency
            yield self.env.timeout(self.wilk_delay)
            # Latency depends on Amplitude and FIFO status (!!!)
            # Analize dynamic range
            self.lost = self.out.put(self.msg,self.lost)



class FE_outlink(object):
    def __init__(self,env,out_stream,param):
        self.env = env
        self.FIFO_out_size = param.FIFO_out_depth
        self.res = simpy.Store(self.env,capacity=self.FIFO_out_size)
        self.action = env.process(self.run())
        self.out_stream = out_stream
        self.latency = int(1E9/param.FE_outrate)
        self.log = np.array([]).reshape(0,2)

    def print_stats(self):
        #print ('TIME: %d // ITEMS: %s ' % (self.env.now,self.res.items))
        self.log=np.vstack([self.log,[len(self.res.items),self.env.now]])

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_out_size):
                self.res.put(data)
                self.print_stats()
                return lost
            else:
                raise Full('OUT LINK FIFO is FULL')
        except Full as e:
            #print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            yield self.env.timeout(self.latency)
            msg = yield self.res.get()
            #self.print_stats()
            self.out_stream.append(msg)



def simulation(run,ch_rate,FE_outrate,FIFO_depth,FIFO_out_depth,FE_ch_latency,
                TE, TGAIN, sensors, events):
    data_out=[]
    lostP,lostC = 0,0
    Param = parameters( ch_rate    = ch_rate,
                        FE_outrate = FE_outrate,   # 2.6Gb/s - 80 bits ch
                        FIFO_depth  = FIFO_depth,
                        FIFO_out_depth = FIFO_out_depth,
                        FE_ch_latency = FE_ch_latency,    # Max Wilkinson Latency
                        TE = TE,
                        TGAIN = TGAIN,
                        sensors = sensors,
                        events = events)

    timing = np.random.randint(0,int(1E9/Param.ch_rate),size=events)

    env = simpy.Environment()

    # System Instanciation and Wiring
    P = [producer(env,
                  data = DATA[:,i],
                  timing  = timing,
                  param = Param) for i in range(Param.sensors)]
    C = [FE_channel(env,
                  param=Param) for i in range(Param.sensors)]
    L = FE_outlink(env,data_out,Param)

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
                'data_out':data_out
                }

    return output



if __name__ == '__main__':

    disk  = hdf_access("/home/viherbos/TEMP/","processed.h5")
    DATA,sensors,events = disk.read()

    runs = 500

    # Multiprocess Work
    pool_size = mp.cpu_count()
    pool = mp.Pool(processes=pool_size)

    sim_info={    'ch_rate'       : 200E3,
                  'FE_outrate'    : (2.6E9/80)/2,
                  # 2.6Gb/s - 80 bits ch
                  'FIFO_depth'    : 4,
                  'FIFO_out_depth': 64*4,
                  'FE_ch_latency' : 5120,
                  # Max Wilkinson Latency
                  'TE' : 5,
                  'TGAIN' : 1,
                  'sensors' : 64,
                  'events' : events}

    mapfunc = partial(simulation, **sim_info)
    pool_output = pool.map(mapfunc, (i for i in range(runs)))

    pool.close()
    pool.join()

    lostP = [pool_output[j]['lostP'] for j in range(runs)]
    lostC = [pool_output[j]['lostC'] for j in range(runs)]
    outlink_ch = [ len(pool_output[j]['data_out']) for j in range(runs)]


    fit = fit_library.gauss_fit()
    fig = plt.figure()
    fit(lostP,'sqrt')
    fit.plot(axis = fig.add_subplot(131),
            title = "FE FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(lostC,'sqrt')
    fit.plot(axis = fig.add_subplot(132),
            title = "Data Link FIFO drops",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)
    fit(outlink_ch,'sqrt')
    fit.plot(axis = fig.add_subplot(133),
            title = "Recovered Channel Data",
            xlabel = "Number of Channel Events",
            ylabel = "Hits",
            res = False)
    plt.show()




#output = mapfunc(1,**sim_info)

# print ("\n --------------------------------- \n")
# print ("FE Input Lost Events %d / Recovered Events %d\n" % (output['lostP'],
#                                                             len(output['data_out'])))
# print ("------------------------------------ \n")
#
# print ("\n --------------------------------- \n")
# print ("FE Output Lost Events %d / Recovered Events %d\n" % (output['lostC'],
#                                                             len(output['data_out'])))
# print ("------------------------------------ \n")
#
# plt.plot(output['log'][:,1],output['log'][:,0])
# plt.show()
