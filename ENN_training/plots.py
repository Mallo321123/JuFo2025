#Filename: plots.py
#A visualisation of what is being simulated. 

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def plot_results(results_t, Aresults_c, Aresults_s, Bresults_c, Bresults_s, Complete_time, av_waiting_time, show):
    # subplots
    fig, (axA, axB) = plt.subplots(nrows=1, ncols=2)
 
    # SignalA
    #
    #
    axSA = axA.twinx()
    axA.plot(results_t, Aresults_c, color='#121613')
    #axSA.plot(Aresults_t, Aresults_w, color='#252D2C')
    axA.set_title(f'Signal light A: Cars waiting per timestep')
    axA.set_xlabel('Passed time since simulation started [timesteps]')
    axA.set_ylabel('Cars waiting at this signal')
    #axSA.set_ylabel('Total waiting time of all cars at this signal [timesteps]')

    #sort periods
    Aresults_t_green = [results_t[i] for i in range(len(results_t)) if Aresults_s[i] == 3]
    Aresults_t_red = [results_t[i] for i in range(len(results_t)) if Aresults_s[i] == 1]

    #plot green periods
    signalcolorG=[]
    for i in range (len(Aresults_t_green)):
        signalcolorG.append(Rectangle((Aresults_t_green[i], axA.get_ylim()[0]),(1), (axA.get_ylim()[1]-axA.get_ylim()[0])))
    pc = PatchCollection(signalcolorG, facecolor='#618B25', alpha=0.3)
    axA.add_collection(pc)

    #plot red periods
    signalcolorR=[]
    for i in range (len(Aresults_t_red)):
        signalcolorR.append(Rectangle((Aresults_t_red[i], axA.get_ylim()[0]),(1), (axA.get_ylim()[1]-axA.get_ylim()[0])))
    pc = PatchCollection(signalcolorR, facecolor='white', alpha=1)
    axA.add_collection(pc)

    # SignalB
    #
    #
    axSB = axB.twinx()
    axB.plot(results_t, Bresults_c, color='#121613')
    #axSB.plot(Aresults_t, Bresults_w, color='#AF0B08')
    axB.set_title(f'Signal light B: Cars waiting per timestep')
    axB.set_xlabel('Passed time since simulation started [timesteps]')
    axB.set_ylabel('Cars waiting at this signal')
    #axSB.set_ylabel('Total waiting time of all cars at this signal [timesteps]')

    #sort periods
    Bresults_t_green = [results_t[i] for i in range(len(results_t)) if Bresults_s[i] == 3]
    Bresults_t_red = [results_t[i] for i in range(len(results_t)) if Bresults_s[i] == 1]

    #plot green periods
    signalcolorG=[]
    for i in range (len(Bresults_t_green)):
        signalcolorG.append(Rectangle((Bresults_t_green[i], axB.get_ylim()[0]),(1), (axB.get_ylim()[1]-axB.get_ylim()[0])))
    pc = PatchCollection(signalcolorG, facecolor='#618B25', alpha=0.3)
    axB.add_collection(pc)

    #plot red periods
    signalcolorR=[]
    for i in range (len(Bresults_t_red)):
        signalcolorR.append(Rectangle((Bresults_t_red[i], axB.get_ylim()[0]),(1), (axB.get_ylim()[1]-axB.get_ylim()[0])))
    pc = PatchCollection(signalcolorR, facecolor='white', alpha=1)
    axB.add_collection(pc)

    if show == True:
        plt.show()
    if show == False:
        plt.close

    #A Histogram of the different waittimes.
    # fig_hist, ax_hist = plt.subplots()
    # ax_hist.hist(av_waiting_time,bins=100, color='#252D2C')
    # ax_hist.set_title(f'Histogram of the average waiting time on all signals')
    # ax_hist.set_ylabel('Average waiting time on all signals [timesteps]')
    # ax_hist.set_xlabel('Number of occurences')

    #plt.show()