#Filename: signal_simulation.py
#Used for the simulation. Shortens the simulation code.
#This file needs to be executed.
from enum import Enum
import numpy as np
from typing import List

#varbiable declarations
globaltime = 0
changetype = 0

#Using Enum to describe the three light stages with numbers.
class SignalState(Enum):
    Red = 1
    Yellow = 2
    Green = 3

#Each waiting car has their own waiting period.
class Car():
    def __init__(self):
        self.t_waiting_period = 0

#To make my simulation more realistic I am using a poisson distribution to randomise certain numbers.
class Signal():
    def __init__(self,startingstate, av_cars_per_timestep, av_cars_crossing_per_timestep):
        self.signal_state = startingstate
        self.t_running_period = 0
        self.t_previous_period = 0
        self.num_switches = 0
        self.cars_per_timestep = av_cars_per_timestep
        self.cars_crossing_per_timestep = av_cars_crossing_per_timestep
        self.rng = np.random.default_rng()
        self.steps_since_last_crossing = 0
        self.step_to_time_conversion = 1
        self.waiting_cars : List[Car]= [Car() for i in range(self.rng.poisson(2))]
        
    def update(self):
        self.addwaitingcars()
        if self.signal_state == SignalState.Green:
            self.increment_running_period()
            self.disappearingcars()
        if self.signal_state == SignalState.Red:
            self.t_running_period = 0
        if self.signal_state == SignalState.Yellow:
            self.t_running_period = 0
        for i in range (len(self.waiting_cars)):
            self.waiting_cars[i].t_waiting_period += 1
        self.total_waitingtime = self.total_wait_time()

    def increment_running_period(self):
        self.t_running_period += 1

    def waitingtime(self):
        for C in self.waiting_cars:
            C.t_waiting_period += 1

    def addwaitingcars(self):
        self.arriving_cars = self.rng.poisson(self.cars_per_timestep)
        for d in range (self.arriving_cars):
            self.waiting_cars.append(Car())

    def disappearingcars(self):
        self.disapearing_cars = self.rng.poisson(self.cars_crossing_per_timestep)
        if self.t_running_period <= self.rng.poisson(2.):
            return
        if (self.steps_since_last_crossing % 1./ self.cars_crossing_per_timestep) < 1e-6:
            try:
                self.waiting_cars.pop(0)
            except IndexError:
                pass
            self.steps_since_last_crossing = 0                   
        self.steps_since_last_crossing +=1 

    #0 --> A green, 1 --> B green
    def signalchangecoordinationA(self, changedecision):
        if changedecision == 0:
            self.signal_state = SignalState.Green
        elif changedecision == 1:
            self.signal_state = SignalState.Red
        elif changetype == 2: #emergency state // all red // end programme
            self.signal_state = SignalState.Red
            raise SystemExit

    def signalchangecoordinationB(self, changedecision):
        if changedecision == 1:
            self.signal_state = SignalState.Green
        elif changedecision == 0:
            self.signal_state = SignalState.Red
        elif changetype == 2: #emergency state // all red // end programme
            self.signal_state = SignalState.Red
            raise SystemExit   

    def total_wait_time(self):
        total = 0
        for c in self.waiting_cars:
            total += c.t_waiting_period
        return total

def get_longest_wait_times(Signals):
    cars = []
    for signal in Signals:
        cars.extend(signal.waiting_cars)
    cars.sort(reverse=True, key = lambda c: c.t_waiting_period)
    if len(cars)<10:
        return [c.t_waiting_period for c in cars] + (10-len(cars))*[.0]
    else:
        return [c.t_waiting_period for c in cars[0:9]]