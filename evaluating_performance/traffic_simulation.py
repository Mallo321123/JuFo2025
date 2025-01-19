import numpy as np
from typing import List
from tqdm import tqdm 
from plots import plot_results
from signal_simulation import SignalState, Signal
from fitness_function import fitness_score

def execute_simulation_nn(traffic_at_A, traffic_at_B, d_traffic_at_A, d_traffic_at_B, steps, neural_network, show):

    #Inizialising all variabels. 
    changetype = 0
    fitness = 0
    avg_fitness = 0
    previous_switch = 0
    current_switch = 0
    switches = 0
    sum_of_waiting_times_A = 0
    sum_of_waiting_times_B = 0

    SignalA = Signal(startingstate=SignalState.Red, av_cars_per_timestep = traffic_at_A, av_cars_crossing_per_timestep = d_traffic_at_A)
    SignalB = Signal(startingstate=SignalState.Green, av_cars_per_timestep= traffic_at_B, av_cars_crossing_per_timestep = d_traffic_at_B)

    #This is needed later to plot and visualise.
    results_t = np.zeros((steps,), dtype=int)
    Aresults_c = np.zeros((steps,), dtype=int)
    Aresults_s = np.zeros((steps,), dtype=SignalState)
    Bresults_c = np.zeros((steps,), dtype=int)
    Bresults_s = np.zeros((steps,), dtype=SignalState)
    Complete_time = np.zeros((steps,), dtype=int)

    avg_signals_waiting_time = []
    fitness=[]

    for i in range(steps):

        #Updating the signal: Adding cars. Removing cars depending on the signal status. Increasing waitingtimes.
    
        SignalA.update() 
        SignalB.update()
        
        signals_waiting_time = SignalA.total_wait_time() + SignalB.total_wait_time()

        avg_signals_waiting_time.append((SignalA.total_wait_time() + SignalB.total_wait_time())/2)
        
        A_waiting_time = SignalA.total_wait_time()
        B_waiting_time = SignalB.total_wait_time()

        time_since_last_switch = 0
        if SignalA.signal_state == SignalState.Green:
            time_since_last_switch = SignalA.t_running_period
        elif SignalB.signal_state == SignalState.Green:
            time_since_last_switch = SignalB.t_running_period

        if previous_switch == current_switch:
            pass
        elif previous_switch != current_switch:
            switches +=1
            previous_switch = current_switch

        current_fitness = fitness_score(signals_waiting_time, A_waiting_time, B_waiting_time, time_since_last_switch, traffic_at_A, traffic_at_B, steps, switches)
        fitness.append(current_fitness)

        #Here the neuronal network is called and given its 3 input values.
        output = neural_network.think([len(SignalA.waiting_cars), len(SignalB.waiting_cars), time_since_last_switch])
        if output[0]>=0:
            changetype = 1
        elif output[0] < 0:
            changetype = 0

        #The Output of the neural network is applied.
        current_switch = changetype
      
        SignalA.signalchangecoordinationA(changetype)
        SignalB.signalchangecoordinationB(changetype)

        #Graphical dipiction & saving the statistical data

        for a in range(len(SignalA.waiting_cars)):
            sum_of_waiting_times_A += SignalA.waiting_cars[a].t_waiting_period

        for b in range(len(SignalB.waiting_cars)):
            sum_of_waiting_times_B += SignalB.waiting_cars[b].t_waiting_period

        Complete_time[i]=signals_waiting_time

        results_t[i] = i
        Aresults_c[i] = len(SignalA.waiting_cars)
        Aresults_s[i] = SignalA.signal_state.value
    
        Bresults_c[i]= len(SignalB.waiting_cars)
        Bresults_s[i]= SignalB.signal_state.value
    
    #Outputting the statistical performance data of the neural network. 
    avg_fitness = np.average(fitness)

    print("ENN-neural network")
    print("----------------------------------------------------")
    print("ENN:Summarised waiting times A:", sum_of_waiting_times_A)
    print("ENN: Summarised waiting times B:", sum_of_waiting_times_B)
    print("Switches", switches)

    if show == True:
        plot_results(results_t, Aresults_c, Aresults_s, Bresults_c, Bresults_s, Complete_time, avg_signals_waiting_time, show)

    return avg_fitness

#Testing the timed algorithm.
def execute_simulation_algt(traffic_at_A, traffic_at_B, d_traffic_at_A, d_traffic_at_B, steps, show):

    changetype = 0
    fitness = 0
    avg_fitness = 0
    previous_switch = 0
    current_switch = 0
    switches = 0
    timer = 0
    sum_of_waiting_times_A = 0
    sum_of_waiting_times_B = 0

    SignalA = Signal(startingstate=SignalState.Red, av_cars_per_timestep = traffic_at_A, av_cars_crossing_per_timestep = d_traffic_at_A)
    SignalB = Signal(startingstate=SignalState.Green, av_cars_per_timestep= traffic_at_B, av_cars_crossing_per_timestep = d_traffic_at_B)

    #This is needed later to plot and visualise.
    results_t = np.zeros((steps,), dtype=int)
    Aresults_c = np.zeros((steps,), dtype=int)
    Aresults_s = np.zeros((steps,), dtype=SignalState)
    Bresults_c = np.zeros((steps,), dtype=int)
    Bresults_s = np.zeros((steps,), dtype=SignalState)
    Complete_time = np.zeros((steps,), dtype=int)

    avg_signals_waiting_time = []
    fitness=[]

    for i in range(steps):
    
        SignalA.update() 
        SignalB.update()
        
        signals_waiting_time = SignalA.total_wait_time() + SignalB.total_wait_time()

        avg_signals_waiting_time.append((SignalA.total_wait_time() + SignalB.total_wait_time())/2)
        
        A_waiting_time = SignalA.total_wait_time()
        B_waiting_time = SignalB.total_wait_time()

        time_since_last_switch = 0
        if SignalA.signal_state == SignalState.Green:
            time_since_last_switch = SignalA.t_running_period
        elif SignalB.signal_state == SignalState.Green:
            time_since_last_switch = SignalB.t_running_period

        if previous_switch == current_switch:
            pass
        elif previous_switch != current_switch:
            switches +=1
            previous_switch = current_switch

        current_fitness = fitness_score(signals_waiting_time, A_waiting_time, B_waiting_time, time_since_last_switch, traffic_at_A, traffic_at_B, steps, switches)
        fitness.append(current_fitness)

        if timer < 30:
            pass
        elif timer == 30:
            changetype = 1
        elif timer == 60:
            changetype = 0
            timer = 0
        else: pass

        timer += 1
        current_switch = changetype
      
        SignalA.signalchangecoordinationA(changetype)
        SignalB.signalchangecoordinationB(changetype)

        #Graphical dipiction & saving the statistical data

        for a in range(len(SignalA.waiting_cars)):
            sum_of_waiting_times_A += SignalA.waiting_cars[a].t_waiting_period

        for b in range(len(SignalB.waiting_cars)):
            sum_of_waiting_times_B += SignalB.waiting_cars[b].t_waiting_period
        Complete_time[i]=signals_waiting_time


        results_t[i] = i
        Aresults_c[i] = len(SignalA.waiting_cars)
        Aresults_s[i] = SignalA.signal_state.value
    
        Bresults_c[i]= len(SignalB.waiting_cars)
        Bresults_s[i]= SignalB.signal_state.value
    
    avg_fitness = np.average(fitness)

    print("Timed algorithm ")
    print("----------------------------------------------------")
    print("Summarised waiting times A:", sum_of_waiting_times_A)
    print("Summarised waiting times B:", sum_of_waiting_times_B)
    print("Switches", switches)

    if show == True:
        plot_results(results_t, Aresults_c, Aresults_s, Bresults_c, Bresults_s, Complete_time, avg_signals_waiting_time, show)

    return avg_fitness

#Testing the timed algorithm plus".
def execute_simulation_algs(traffic_at_A, traffic_at_B, d_traffic_at_A, d_traffic_at_B, steps, show):

    changetype = 0
    fitness = 0
    avg_fitness = 0
    previous_switch = 0
    current_switch = 0
    switches = 0
    timer = 0
    sum_of_waiting_times_A = 0
    sum_of_waiting_times_B = 0

    SignalA = Signal(startingstate=SignalState.Red, av_cars_per_timestep = traffic_at_A, av_cars_crossing_per_timestep = d_traffic_at_A)
    SignalB = Signal(startingstate=SignalState.Green, av_cars_per_timestep= traffic_at_B, av_cars_crossing_per_timestep = d_traffic_at_B)

    #This is needed later to plot and visualise.
    results_t = np.zeros((steps,), dtype=int)
    Aresults_c = np.zeros((steps,), dtype=int)
    Aresults_s = np.zeros((steps,), dtype=SignalState)
    Bresults_c = np.zeros((steps,), dtype=int)
    Bresults_s = np.zeros((steps,), dtype=SignalState)
    Complete_time = np.zeros((steps,), dtype=int)

    avg_signals_waiting_time = []
    fitness=[]

    for i in range(steps):
    
        SignalA.update() 
        SignalB.update()
        
        signals_waiting_time = SignalA.total_wait_time() + SignalB.total_wait_time()

        avg_signals_waiting_time.append((SignalA.total_wait_time() + SignalB.total_wait_time())/2)
        
        A_waiting_time = SignalA.total_wait_time()
        B_waiting_time = SignalB.total_wait_time()

        time_since_last_switch = 0
        if SignalA.signal_state == SignalState.Green:
            time_since_last_switch = SignalA.t_running_period
        elif SignalB.signal_state == SignalState.Green:
            time_since_last_switch = SignalB.t_running_period

        if previous_switch == current_switch:
            pass
        elif previous_switch != current_switch:
            switches +=1
            previous_switch = current_switch

        current_fitness = fitness_score(signals_waiting_time, A_waiting_time, B_waiting_time, time_since_last_switch, traffic_at_A, traffic_at_B, steps, switches)
        fitness.append(current_fitness)

        waiting_at_A = len(SignalA.waiting_cars)
        waiting_at_B = len(SignalB.waiting_cars)

        #Implementation of the switching algorithm.
        if timer < 20:
            if waiting_at_A == 0:
                changetype = 1
            elif waiting_at_B == 0:
                changetype = 0
            else: 
                timer += 1
        else: 
            if changetype == 0:
                changetype = 1
            elif changetype == 1:
                changetype = 0
            timer = 0

        current_switch = changetype
      
        SignalA.signalchangecoordinationA(changetype)
        SignalB.signalchangecoordinationB(changetype)

        #Graphical dipiction & saving the statistical data

        for a in range(len(SignalA.waiting_cars)):
            sum_of_waiting_times_A += SignalA.waiting_cars[a].t_waiting_period

        for b in range(len(SignalB.waiting_cars)):
            sum_of_waiting_times_B += SignalB.waiting_cars[b].t_waiting_period

        Complete_time[i]=signals_waiting_time

        results_t[i] = i
        Aresults_c[i] = len(SignalA.waiting_cars)
        Aresults_s[i] = SignalA.signal_state.value
    
        Bresults_c[i]= len(SignalB.waiting_cars)
        Bresults_s[i]= SignalB.signal_state.value
    
    avg_fitness = np.average(fitness)
    print("Timed algorithm plus")
    print("----------------------------------------------------")
    print("Summarised waiting times A:", sum_of_waiting_times_A)
    print("Summarised waiting times B:", sum_of_waiting_times_B)
    print("Switches:", switches)

    if show == True:
        plot_results(results_t, Aresults_c, Aresults_s, Bresults_c, Bresults_s, Complete_time, avg_signals_waiting_time, show)

    return avg_fitness