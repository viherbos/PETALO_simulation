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

    def __init__(self,env,max_delay,FIFO_size,message):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())
        self.counter = 0
        self.data=message
        self.max_delay = max_delay
        self.max_FIFO = FIFO_size
        self.lost = 0

    def run(self):
        while self.counter < len(self.data):
            yield self.env.timeout(random.randint(1,self.max_delay))
            #print_stats(env,self.out.res)

            try:
                if (len(self.out.res.items)<self.max_FIFO):
                    self.out.put(self.data[self.counter])
                    self.counter += 1
                else:
                    self.counter += 1
                    raise Full('FIFO is FULL')
                    # Drop data. FIFO is FULL so data is lost
            except IndexError:
                print "List Empty"
            except Full as e:
                print ("TIME: %s // %s" % (self.env.now,e.value))
                self.lost += 1




class consumer(object):

    def __init__(self,env,delay,data_out):
        self.env = env
        self.res = simpy.Store(self.env,capacity=2)
        self.action = env.process(self.run())
        self.data_out = data_out
        self.delay = delay
        self.index = 0

    def print_stats(self):
        print ('TIME: %d // ITEMS: %s ' % (self.env.now,self.res.items))

    def put(self,data):
        self.res.put(data)
        self.print_stats()

    def run(self):
        while True:
            yield self.env.timeout(random.randint(1,self.delay))
            msg = yield self.res.get()
            self.data_out.append(msg)
            self.index += 1


if __name__ == '__main__':

    lost_vector1=np.zeros(1000)
    lost_vector2=np.zeros(1000)

    for i in range(1000):
        message1 = np.random.randint(1,10,size=100)
        message2 = np.random.randint(1,10,size=100)
        env = simpy.Environment()
        data_out=[]
        C = consumer(env,
                    delay=5,
                    data_out=data_out)

        P1 = producer(env,
                    max_delay=2,
                    FIFO_size=2,
                    message = message1)

        P2 = producer(env,
                    max_delay=2,
                    FIFO_size=2,
                    message = message1)

        P1.out = C
        P2.out = C

        env.run()
        print ("\n -------------------- \n \
Total Lost Events %d \n --------------------" % P1.lost)
        lost_vector1[i] = P1.lost

        print ("\n -------------------- \n \
Total Lost Events %d \n --------------------" % P2.lost)
        lost_vector2[i] = P2.lost

    fit = fit_library.gauss_fit()
    fig = plt.figure()

    fit(lost_vector1,'sqrt')
    fit.plot(axis = fig.add_subplot(111),
            title = "Lost Events Histogram",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)

    fit(lost_vector2,'sqrt')
    fit.plot(axis = fig.add_subplot(111),
            title = "Lost Events Histogram",
            xlabel = "Number of Lost Events",
            ylabel = "Hits",
            res = False)

    plt.show()
