import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_analysis/")
import fit_library


class Full(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class producer(object):

    def __init__(self,env,max_delay,message):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())
        self.counter = 0
        self.data=message
        self.max_delay = max_delay
        self.lost = 0

    def run(self):
        while self.counter < len(self.data):
            yield self.env.timeout(random.randint(1,self.max_delay))
            #print_stats(env,self.out.res)

            try:
                self.lost = self.out.put(self.data[self.counter],self.lost)
                self.counter += 1
                # Drop data. FIFO is FULL so data is lost
            except IndexError:
                print "List Empty"





class consumer(object):

    def __init__(self,env,delay,FIFO_size,data_out):
        self.env = env
        self.FIFO_size = FIFO_size
        self.res = simpy.Store(self.env,capacity=self.FIFO_size)
        self.action = env.process(self.run())
        self.data_out = data_out
        self.delay = delay
        self.index = 0

    def print_stats(self):
        print ('TIME: %d // ITEMS: %s ' % (self.env.now,self.res.items))

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_size):
                self.res.put(data)
                return lost
            else:
                raise Full('FIFO is FULL')
        except Full as e:
            #print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)
        # self.print_stats()

    def run(self):
        while True:
            yield self.env.timeout(random.randint(1,self.delay))
            msg = yield self.res.get()
            self.data_out.append(msg)
            self.index += 1


if __name__ == '__main__':

    iterations = 500

    lost_vector1=np.zeros(iterations)
    lost_vector2=np.zeros(iterations)


    for j in range(iterations):
        message1 = np.random.randint(1,10,size=100)
        message2 = np.random.randint(1,10,size=100)
        env = simpy.Environment()
        data_out=[]
        C = consumer(env,
                    delay=5,
                    FIFO_size=2,
                    data_out=data_out)

        P = [producer(env,
                     max_delay=2,
                     message = message1) for i in range(64)]

        for i in range(64):
            P[i].out = C


        env.run()
        lost_vector1[j] = P[0].lost
        lost_vector2[j] = P[1].lost
        
#         print ("\n ------------------------ \n \
# Total Lost Events %d from P1 \n ------------------------" % P[0].lost)
#
#         print ("\n ------------------------ \n \
# Total Lost Events %d from P2\n ------------------------" % P[1].lost)


    fit = fit_library.gauss_fit()
    fig = plt.figure()

    fit(lost_vector1,'sqrt')
    fit.plot(axis = fig.add_subplot(121),
            title = "Lost Events Histogram",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)

    fit(lost_vector2,'sqrt')
    fit.plot(axis = fig.add_subplot(122),
            title = "Lost Events Histogram",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)

    plt.show()
