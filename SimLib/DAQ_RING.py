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
                 FE_ch_latency,TE,TGAIN,FE_n_channels,
                 sensors,events,L1_outrate,FIFO_L1a_depth,FIFO_L1b_depth,
                 n_asics):
        self.ch_rate     = ch_rate
        self.FE_outrate  = FE_outrate
        self.FIFO_depth  = FIFO_depth
        self.sensors     = sensors
        self.events      = events
        self.FIFO_out_depth = FIFO_out_depth
        self.FE_ch_latency  = FE_ch_latency      # Maximum Ch Latency
        self.TE         = TE
        self.TGAIN      = TGAIN
        self.FE_n_channels = FE_n_channels
        self.L1_outrate = L1_outrate
        self.FIFO_L1a_depth = FIFO_L1a_depth
        self.FIFO_L1b_depth = FIFO_L1b_depth
        self.n_asics = n_asics


class ch_frame(object):

    def __init__(self,data,event,sensor_id,asic_id,in_time,out_time):
        self.data = data
        self.sensor_id = sensor_id
        self.event = event
        self.asic_id = asic_id
        self.in_time = in_time
        self.out_time = out_time

    def get_np_array(self):
        return np.array([self.data, self.event, self.sensor_id, self.asic_id,
                self.in_time, self.out_time])

    # def put_np_array(self, nparray):
    #     aux_list = {'data'      :   nparray[0],
    #                 'event'     :   nparray[1],
    #                 'sensor_id' :   nparray[2],
    #                 'asic_id'   :   nparray[3],
    #                 'in_time'   :   nparray[4],
    #                 'out_time'  :   nparray[5]}
    #
    #     self.data       = aux_list['data']
    #     self.sensor_id  = aux_list['sensor_id']
    #     self.event      = aux_list['event']
    #     self.asic_id    = aux_list['asic_id']
    #     self.in_time    = aux_list['in_time']
    #     self.out_time   = aux_list['out_time']

    def __repr__(self):
        return "data: {}, event: {}, sensor_id: {}, asic_id: {} in_time:{} out_time:{}".\
            format( self.data, self.event, self.sensor_id, self.asic_id,
                    self.in_time, self.out_time)


class L1_outframe(object):

    def __init__(self,data,event,asic_id,in_time,out_time):
        self.data = data
        # Lenght of data is not constant, depends on the number of channels being sent
        # DATA fiels: n_CH | TDC | SENSOR1 | QDC1 | SENSOR2 | QDC2 | ... | B_QDC
        self.event = event
        self.asic_id = asic_id
        self.in_time = in_time
        self.out_time = out_time

    def get_np_array(self):
        B = [ self.event, self.asic_id, self.in_time, self.out_time]
        return np.concatenate((self.data,B),axis=0)

    def __repr__(self):
        return "data: {}, event: {}, asic_id: {} in_time:{} out_time:{}".\
            format(self.data,self.event,self.asic_id,self.in_time,self.out_time)



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

    def __init__(self,env,data,timing,param,sensor_id,asic_id):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())
        self.counter = 0
        self.lost = 0
        self.data = data
        self.timing = timing
        self.TE = param.TE
        self.sensor_id = sensor_id
        self.asic_id = asic_id


    def run(self):
        while self.counter < len(self.data):

            yield self.env.timeout(int(self.timing[self.counter]))
            #print_stats(env,self.out.res)

            try:
                if self.data[self.counter]>self.TE:
                    self.DATA = ch_frame(data     = self.data[self.counter],
                                        event     = self.counter,
                                        sensor_id = self.sensor_id,
                                        asic_id   = self.asic_id,
                                        in_time   = self.env.now,
                                        out_time  = 0)
                    #np.array([self.data[self.counter],self.env.now,0])
                    # PACKET FRAME: [SENSOR_DATA, IN_TIME, OUT_TIME]
                    self.lost = self.out.put(self.DATA.get_np_array(),self.lost)
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

    def __init__(self,env,param,sensor_id):
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
        self.sensor_id = sensor_id

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
                raise Full('Channel FIFO is FULL')
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

    def __init__(self,env,param,asic_id):
        self.env = env
        self.FIFO_out_size = param.FIFO_out_depth
        self.res = simpy.Store(self.env,capacity=self.FIFO_out_size)
        self.action = env.process(self.run())
        self.latency = int(1E9/param.FE_outrate)
        self.log = np.array([]).reshape(0,2)
        self.asic_id = asic_id
        self.out = None
        self.lost = 0

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
            self.packet = yield self.res.get()
            yield self.env.timeout(self.latency)
            self.lost = self.out.put(self.packet,self.lost)



class FE_asic(object):
    """ ASIC model.
        Method

        Parameters
        sensor_id : Array with the positions of the sensors being used (param.sensors)
    """
    def __init__(self,env,param,data,timing,sensors,asic_id):
        self.env        = env
        self.Param      = param
        self.DATA       = data
        self.timing     = timing
        self.sensors    = sensors
        self.asic_id    = asic_id
        self.n_ch       = param.FE_n_channels


        # System Instanciation and Wiring
        self.Producer = [producer(   self.env,
                                data       = self.DATA[:,i],
                                timing     = self.timing,
                                param      = self.Param,
                                sensor_id  = self.sensors[i],
                                asic_id    = self.asic_id)
                                            for i in range(self.n_ch)]
        self.Channels = [FE_channel( self.env,
                                param = self.Param,
                                sensor_id = self.sensors[i])
                                            for i in range(self.n_ch)]
        self.Link     = FE_outlink(  self.env,
                                self.Param,
                                asic_id = self.asic_id)

        for i in range(self.n_ch):
            self.Producer[i].out = self.Channels[i]
            self.Channels[i].out = self.Link



class L1(object):
    """ L1 model.
        Methods

        Parameters

    """
    def __init__(self,env,out_stream,param,L1_id):
        self.env        = env
        self.Param      = param
        self.L1_id      = L1_id
        self.out_stream = out_stream
        self.latency    = int(1E9/param.L1_outrate)
        self.res        = simpy.Store(self.env,capacity=param.FIFO_L1a_depth)
        self.res_out    = simpy.Store(self.env,capacity=param.FIFO_L1b_depth)
        self.action     = env.process(self.run())


    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.Param.FIFO_L1_depth):
                self.res.put(data)
                #self.print_stats()
                return lost
            else:
                raise Full('L1 FIFO is FULL')
        except Full as e:
            #print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            yield self.env.timeout(self.latency)
            self.msg = yield self.res.get()
            self.msg[5] = self.env.now
            #self.print_stats()
            self.out_stream = np.pad(self.out_stream,((1,0),(0,0)),
                                    mode='constant',
                                    constant_values=0)
            self.out_stream[0,:] = self.msg



class L1_hierarchy(object):


    def __init__(self,env,param,DATA,timing,sensors,L1_id):
        self.env        = env
        self.param      = param
        self.data_out   = np.array([]).reshape(0,6)
        self.block_size = param.FE_n_channels

        self.L1    = L1( env        = self.env,
                         out_stream = self.data_out,
                         param      = self.param,
                         L1_id      = L1_id)

        self.ASICS = [FE_asic( self.env,
                               param = self.param,
                               data  = DATA[:,i*self.block_size:(i+1)*self.block_size],
                               timing = timing,
                               sensors = sensors[i*self.block_size:(i+1)*self.block_size],
                               asic_id = i)
                               for i in range(self.param.n_asics)]

        for i in range(self.param.n_asics):
            self.ASICS[i].Link.out = self.L1


    def __call__(self):
        lostP = np.array([]).reshape(0,1)
        lostC = np.array([]).reshape(0,1)
        lostL1 = np.array([]).reshape(0,1)

        for j in range(self.param.n_asics):
            for i in range(self.param.FE_n_channels):
                lostP = np.pad(lostP,((1,0),(0,0)), mode='constant', constant_values=0)
                lostC = np.pad(lostC,((1,0),(0,0)), mode='constant', constant_values=0)
                lostP[0,:] = self.ASICS[j].Producer[i].lost
                lostC[0,:] = self.ASICS[j].Channels[i].lost
                #lostP = lostP + self.ASICS[j].Producer[i].lost
                #lostC = lostC + self.ASICS[j].Channels[i].lost
            lostL1 = np.pad(lostL1,((1,0),(0,0)), mode='constant', constant_values=0)
            lostL1[0,:] = self.ASICS[j].Link.lost
            #lostL1 = lostL1 + self.ASICS[j].Link.lost

        output = {  'lostP':lostP,
                    'lostC':lostC,
                    'lostL1':lostL1,
                    'data_out':self.L1.out_stream
                    #'log':ASIC[0].Link.log
                    }

        return output
