import simpy
import random
from simpy.events import AnyOf, AllOf, Event

lost_events_counter = 0

class FullError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        lost_events_counter++
        return repr(self.value)

def print_stats(env,res):
    #print('%d of %d slots are allocated.' % (res.count, res.capacity))
    print ('TIME: %d // %d of %d slots are allocated.' % (env.now,res.count, res.capacity))
    #print('  Users:', res.users)
    #print('  Queued events:', res.queue)

def consumer(env, resource):
    yield env.timeout(random.randint(1,10))
    
    print_stats(env,resource)

    try:
        if resource.count == 2:
             raise FullError('Lost event')
        with resource.request() as req:
            #yield req
            #print_stats(env,resource)
            yield env.timeout(random.randint(1,5))
            # Automatic release of the resource
    except FullError:
        print ("Lost Event at %d" % env.now)


if __name__ == '__main__':

    env = simpy.Environment()
    cola = simpy.Resource(env,capacity=2)
    consumers = [env.process(consumer(env,cola)) for i in range(5)]
    env.run(until=100)
    print ("Total Lost Events %d" % lost_events_counter)
    # for i in range(1,100):
    #     env.run(until=i)
    #     print_stats(env,cola)
