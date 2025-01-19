#Filename: fitness_function.py
#Here I am defining my fitness function depending on some parameters, 
#which should describe the performance of my current running model.

import numpy as np

#Through adjusting the weight of the parameters the switching behaivor could be changed. 
#The parameters alpha, beta, gamma, delta also serve as an scaling factor.
def fitness_score(avg_current_wait_t,waiting_A, waiting_B, switch_time, traffic_at_A, traffic_at_B, steps, switches, alpha=6.0, beta=0.05, gamma=8.0, delta=1):

   fitness = 0
   #The equality is supposed to discribe the percentual ratio between the 
   #waiting cars at side A and side B.
   #If the distribution is equal the score is the highest.
   equality = 0
   if waiting_A > waiting_B:
      equality = (waiting_B / waiting_A)
   elif waiting_B > waiting_A:
      equality =  (waiting_A / waiting_B)
   elif waiting_A == waiting_B:
      equality = 1

   switches_per_runntime = 0
   if steps > 0:
      switches_per_runntime = switches / steps
   else: switches_per_runntime = 0

   switch_time_per_switches = 0
   if switches > 0:
      switch_time_per_switches = switch_time/ switches
   else: switch_time_per_switches = 0

   #The fitness describes how good the model fits for the task. Meaning the higher the fitness, the better the model. 
   #This is my fitness function, surely improvements can be made. During training the fitness function should be modified 
   #to achieve better training results.
   if (avg_current_wait_t * (traffic_at_A + traffic_at_B)) == 0:
      fitness = (beta * switch_time_per_switches + gamma * equality - delta * switches_per_runntime)
   else:
      fitness = (alpha * (1 / (avg_current_wait_t * (traffic_at_A + traffic_at_B))) + beta * switch_time_per_switches + gamma * equality - delta * switches_per_runntime)

   return fitness