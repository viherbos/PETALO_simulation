import simpy

def clock(env,name,tick):
    while True:
        print (name,env.now)
        yield env.timeout(tick)

def main_clock():
    env = simpy.Environment()
    env.process(clock(env,'fast',0.5))
    env.process(clock(env,'slow',1))
    env.run(until=5)

# Complex class

class car(object):
    def __init__(self,env,charge_dur,trip_dur):
        self.env = env
        # Start the run process when instance created
        self.action = env.process(self.run())
        self.charge_dur = charge_dur
        self.trip_dur = trip_dur
        self.charge_ends = self.env.event()

    def run(self):
        while True:
            print("Start driving at %d" % self.env.now)
            yield self.env.timeout(self.trip_dur)

            print("Start parking and charging at %d" % self.env.now)
            yield self.env.process(self.charge())

    def charge(self):
        yield self.env.timeout(self.charge_dur) | self.charge_ends
        # Waits for complete charge or driver interrupt
        if self.charge_ends.triggered:
            self.charge_ends = self.env.event()

class driver(object):
    def __init__(self,env,driver_dur,car):
        self.env = env
        self.driver_dur = driver_dur
        self.car = car

    def driver_i(self):
        yield self.env.timeout(self.driver_dur)
        self.car.charge_ends.succeed()


if __name__ == "__main__":
    env = simpy.Environment()

    CAR = car(env,charge_dur=10,trip_dur=3)
    DRIVER = driver(env,driver_dur=6,car=CAR)

    env.process(DRIVER.driver_i())

    env.run(until=50)
