import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
import HF_translator as HFT
import os
import pandas as pd


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


class frame(object):

    def __init__(self,data,event,sensor_id,asic_id,in_time,out_time):
        self.data = data
        self.sensor_id = sensor_id
        self.event = event
        self.asic_id = asic_id
        self.in_time = in_time
        self.out_time = out_time

    def __repr__(self):
        return "data: {}, event: {}, sensor_id: {}, asic_id: {} in_time:{} out_time:{}".\
            format( self.data, self.event, self.sensor_id, self.asic_id,
                    self.in_time, self.out_time)


class hdf_access(object):
    """ A utility class to access data in hf5 format.
        read method is used to load data from a preprocessed file.
        The file format is a table with each column is a sensor and
        each row an event
    """

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
    """ Sends data to a given channel. DATA has 3 elements:
            Charge, in_time, out_time(0)
        Parameters
        env     : Simpy environment
        counter : Event counter
        lost    : FIFO drops counter (Channel Input FIFO)
        TE      : Energy threshold for channel filtering
        timing  : reads delay from previously generated vector
    """

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
                    self.DATA = np.array([self.data[self.counter],self.env.now,0])
                    # PACKET FRAME: [SENSOR_DATA, IN_TIME, OUT_TIME]
                    self.lost = self.out.put(self.DATA,self.lost)
                self.counter += 1
                # Drop data. FIFO is FULL so data is lost
            except IndexError:
                print "List Empty"


class FE_channel(object):
    """ ASIC channel model.
        Method
        put     : Input FIFO storing method
        Parameters
        env     : Simpy environment
        FIFO_size : Size of input FIFO (4)
        lost    : FIFO drops counter (output FIFO)
        gain    : channel QDC gain
        timing  : reads delay from previously generated vector
        latency : Wilkinson ADC latency (in terms of amplitude)
        log     : Stores log of items and time in input FIFO
    """

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
            self.packet = yield self.res.get()
            self.msg = self.packet[0]
            self.wilk_delay = int((self.latency/1024)*self.msg*self.gain)
            if self.wilk_delay > self.latency:
                self.wilk_delay = self.latency
            yield self.env.timeout(self.wilk_delay)
            # Latency depends on Amplitude and FIFO status (!!!)
            # Analize dynamic range
            self.lost = self.out.put(self.packet,self.lost)



class FE_outlink(object):
    """ ASIC Outlink model.
        Method
        put     : Output link FIFO storing method
        Parameters
        env     : Simpy environment
        FIFO_out_size : Size of output FIFO
        latency : Latency depends on output link speed
        log     : Stores time and number of FIFO elements
    """
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
        # FIFO Statistics

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_out_size):
                self.res.put(data)
                #self.print_stats()
                return lost
            else:
                raise Full('OUT LINK FIFO is FULL')
        except Full as e:
            #print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            yield self.env.timeout(self.latency)
            self.msg = yield self.res.get()
            self.msg[2] = self.env.now
            #self.print_stats()
            self.out_stream = np.vstack([self.out_stream,self.msg])
            #print self.out_stream.shape
