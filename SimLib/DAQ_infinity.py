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

""" LIBRARY FOR INFINITY DAQ """

class Full(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class parameters(object):

    def __init__(self,data,sensors,n_events):
        self.P          = data
        self.sensors    = sensors
        self.events     = n_events
        # ch_rate    = CG.data['ENVIRONMENT']['ch_rate'],
        # FE_outrate = CG.data['TOFPET']['outlink_rate'],
        # FIFO_depth  = CG.data['TOFPET']['IN_FIFO_depth'],
        # FIFO_out_depth = CG.data['TOFPET']['OUT_FIFO_depth'],
        # FE_ch_latency = CG.data['TOFPET']['MAX_WILKINSON_LATENCY'],
        # TE = CG.data['TOFPET']['TE'],
        # TGAIN = CG.data['TOFPET']['TGAIN'],
        # FE_n_channels = CG.data['TOFPET']['n_channels'],
        # sensors = sensors,
        # events = n_events,
        # L1_outrate = CG.data['L1']['L1_outrate'],
        # FIFO_L1a_depth = CG.data['L1']['FIFO_L1a_depth'],
        # FIFO_L1b_depth = CG.data['L1']['FIFO_L1b_depth'],
        # n_asics = CG.data['L1']['n_asics'],
        # TEL1 = CG.data['L1']['TE']


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

    def put_np_array(self, nparray):
        aux_list = {'data'      :   nparray[0], 'event'     :   nparray[1],
                    'sensor_id' :   nparray[2], 'asic_id'   :   nparray[3],
                    'in_time'   :   nparray[4], 'out_time'  :   nparray[5]}

        self.data       = aux_list['data']
        self.sensor_id  = aux_list['sensor_id']
        self.event      = aux_list['event']
        self.asic_id    = aux_list['asic_id']
        self.in_time    = aux_list['in_time']
        self.out_time   = aux_list['out_time']

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

    def get_dict(self):
        return {'data'      :self.data,
        # DATA fiels: n_CH | TDC | SENSOR1 | QDC1 | SENSOR2 | QDC2 | ... | B_QDC
                'event'     :self.event,
                'asic_id'   :self.asic_id,
                'in_time'   :self.in_time,
                'out_time'  :self.out_time}

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
        self.TE = param.P['TOFPET']['TE']
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
        self.FIFO_size = param.P['TOFPET']['IN_FIFO_depth']
        self.res = simpy.Store(self.env,capacity=self.FIFO_size)
        self.action = env.process(self.run())
        self.out = None
        self.latency = param.P['TOFPET']['MAX_WILKINSON_LATENCY']
        self.index = 0
        self.lost = 0
        self.gain = param.P['TOFPET']['TGAIN']
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
        put             : Output link FIFO storing method

        Parameters
        env             : Simpy environment
        FIFO_out_size   : Size of output FIFO
        latency         : Latency depends on output link speed
        log             : Stores time and number of FIFO elements
    """

    def __init__(self,env,param,asic_id):
        self.env = env
        self.FIFO_out_size = param.P['TOFPET']['OUT_FIFO_depth']
        self.res = simpy.Store(self.env,capacity=self.FIFO_out_size)
        self.action = env.process(self.run())
        self.latency = int(1E9/param.P['TOFPET']['outlink_rate'])
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
            yield self.env.timeout(self.latency)
            packet = yield self.res.get()
            self.out.put(packet,self.lost)




class FE_asic(object):
    """ ASIC model.
        Method

        Parameters
        sensor_id : Array with the positions of the sensors being used (param.sensors)
    """
    def __init__(self,env,param,data,timing,sensors,asic_id):
        self.env        = env
        self.param      = param
        self.DATA       = data
        self.timing     = timing
        self.sensors    = sensors
        self.asic_id    = asic_id
        self.n_ch       = param.P['TOFPET']['n_channels']


        # System Instanciation and Wiring
        self.Producer = [producer(   self.env,
                                data       = self.DATA[:,i],
                                timing     = self.timing,
                                param      = self.param,
                                sensor_id  = self.sensors[i],
                                asic_id    = self.asic_id)
                                            for i in range(self.n_ch)]
        self.Channels = [FE_channel( self.env,
                                param = self.param,
                                sensor_id = self.sensors[i])
                                            for i in range(self.n_ch)]
        self.Link     = FE_outlink(  self.env,
                                self.param,
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
        self.L1_id      = L1_id
        self.param      = param
        self.out_stream = out_stream
        self.latency    = int(1E9/param.P['L1']['L1_outrate'])
        self.fifoA      = simpy.Store(self.env,
                                    capacity=param.P['L1']['FIFO_L1a_depth'])
        self.fifoB      = simpy.Store(self.env,
                                    capacity=param.P['L1']['FIFO_L1b_depth'])
        self.action1    = env.process(self.runA())
        self.action2    = env.process(self.runB())
        self.buffer     = np.array([]).reshape(0,6)
        self.flag       = False
        self.frame_count = 0
        self.lostB      = 0
        self.lostA      = 0


    def process_frames(self):
        out=[]
        while (self.buffer.shape[0]>0):
            time = self.buffer[0,5]
            cond = np.array(self.buffer[:,5]==time)
            buffer_sel = self.buffer[cond,:]
            #Select those with same IN_TIME

            data_frame = [-1,time]
            sum_QDC = 0
            n_ch = 0
            for i in buffer_sel[:,:]:
                if i[0] > self.param.P['L1']['TE']:
                    data_frame.append(i[2])
                    data_frame.append(i[0])
                    n_ch +=1
                else:
                    sum_QDC += i[0]
            data_frame.append(sum_QDC)
            data_frame[0] = n_ch
            # Build Data frame

            out.append({'data'      :data_frame,
                        'event'     :self.buffer[0,1],
                        'asic_id'   :self.buffer[0,2],
                        'in_time'   :time,
                        'out_time'  :0})

            #take all the used data out of the buffer
            cond_not = np.invert(cond)
            self.buffer = self.buffer[cond_not]

        return out


    def put(self,data,lost):
        try:
            if (len(self.fifoA.items)<(16)):
                self.fifoA.put(data)
            else:
                raise Full('L1 FIFO A is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            lost += 1
            return lost



    def runA(self):
        while True:
            # if (len(self.fifoA.items) > 8):
            #     self.flag = True
            frame = yield self.fifoA.get()
            yield self.env.timeout(1)

            if (self.frame_count < 9):
                self.buffer = np.pad(self.buffer,((1,0),(0,0)),mode='constant')
                self.buffer[0,:] = frame
                self.frame_count += 1

            if (self.frame_count == 9):
                out = self.process_frames()

                try:
                    if (len(self.fifoB.items)<16):
                        self.putB(out,self.lostB)
                    else:
                        raise Full('L1 FIFO B is FULL')
                except Full as e:
                    print ("TIME: %s // %s" % (self.env.now,e.value))
                    self.lostB += 1

                self.frame_count = 0
                #self.flag = False
                self.buffer = np.array([]).reshape(0,6)


    def putB(self,data,lost):
        try:
            if (len(self.fifoB.items)<(16)):
                self.fifoB.put(data)
            else:
                raise Full('L1 FIFO B is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            lost += 1
            return lost


    def runB(self):
        while True:
            msg = yield self.fifoB.get()
        # 8 bits n_CH | 10 bits TDC | n_CH * (16 bits + 10 bits) | 8 bits B_QDC
            delay = (msg[0]['data'][0]*26 + 8 + 10 + 8)*(1.0/self.param.P['L1']['L1_outrate'])

            yield self.env.timeout(int(delay))
            msg[0]['out_time'] = self.env.now
            self.out_stream.append(msg)


class L1_hierarchy(object):

    def __init__(self,env,param,DATA,timing,sensors,L1_id):
        self.env        = env
        self.param      = param
        self.data_out   = [] #np.array([]).reshape(0,6)
        self.block_size = param.P['TOFPET']['n_channels']

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
                               for i in range(self.param.P['L1']['n_asics'])]

        for i in range(self.param.P['L1']['n_asics']):
            self.ASICS[i].Link.out = self.L1


    def __call__(self):
        lostP = np.array([]).reshape(0,1)
        lostC = np.array([]).reshape(0,1)
        lostL1a = np.array([]).reshape(0,1)
        lostL1b = np.array([]).reshape(0,1)

        for j in range(self.param.P['L1']['n_asics']):
            for i in range(self.param.P['TOFPET']['n_channels']):
                lostP   = np.pad(lostP,((1,0),(0,0)),
                                    mode='constant', constant_values=0)
                lostC   = np.pad(lostC,((1,0),(0,0)),
                                    mode='constant', constant_values=0)
                lostL1a = np.pad(lostL1a,((1,0),(0,0)),
                                    mode='constant', constant_values=0)
                lostL1b = np.pad(lostL1b,((1,0),(0,0)),
                                    mode='constant', constant_values=0)

                lostP[0,:] = self.ASICS[j].Producer[i].lost
                # Lost in CH input FIFO
                lostC[0,:] = self.ASICS[j].Channels[i].lost
                # Lost in ASIC OutLink FIFO
                lostL1a[0,:] = self.L1.lostA
                lostL1b[0,:] = self.L1.lostA

        output = {  'lostP':lostP,
                    'lostC':lostC,
                    'lostL1a':lostL1a,
                    'lostL1b':lostL1b,
                    'data_out':self.L1.out_stream
                    }

        return output
