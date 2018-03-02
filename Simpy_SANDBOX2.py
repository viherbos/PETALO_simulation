import simpy
import random
from simpy.events import AnyOf, AllOf, Event



class FullError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)





class producer(object):

    def __init__(self,env):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())

    def run(self):
        while self.env.now < 100:
            yield self.env.timeout(random.randint(1,10))
            #print_stats(env,self.out.res)
            self.out.put()


class consumer(object):

    def __init__(self,env,lost_ev):
        self.env = env
        self.res = simpy.Resource(self.env,capacity=2)
        self.lost = lost_ev

    def print_stats(self):
        print ('TIME: %d // %d of %d slots are allocated.' % (self.env.now,\
                                                self.res.count, self.res.capacity))

    def put(self):
        #self.print_stats(self.env,self.res)
        print ("HOLA")
        with self.res.request() as req:
             # Automatic release of the resource
             yield self.env.timeout(random.randint(1,5))

        # try:
        #     print_stats(env,self.res)
        #     if self.res.count == 2:
        #         raise FullError('Lost event')
        #     with self.res.request() as req:
        #         # Automatic release of the resource
        #         yield self.env.timeout(random.randint(1,5))
        # except FullError:
        #     print ("Lost Event at %d" % self.env.now)
        #     self.lost += 1


if __name__ == '__main__':

    env = simpy.Environment()
    lost_counter = 0
    C = consumer(env,lost_counter)
    P = producer(env)
    P.out = C

    env.run(until=100)
    print ("Total Lost Events %d" % lost_counter)
