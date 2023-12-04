# Import routines

import numpy as np
import math
import random


# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        """ action_ space : pick up location , Drop location
            state_space   : location , time (hours) , day
             state_init   : random pick from the state_space """
        self.action_space = [(i,j) for i in range(m) for j in range(m) if i!=j or i==0]
        # Total states (Xi Tj Dk)
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        #  random Initialize of state (location, hours, day)
        self.state_init = random.choice(self.state_space)
        # Start the first round
        self.reset()



    def state_encod_arch1(self, state):

        # creating the vector of size 5+24+7=36
        state_encod = np.zeros(m+t+d)
        #storing index of current state,time and day
        state_index=state[0]
        time_index = m + state[1]
        day_index = m + t + state[2]
         # encode location
        state_encod[state_index] = 1
         # encode hour of the day
        state_encod[time_index] = 1
         # encode day of the week
        state_encod[day_index] = 1

        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        #  poisson distribution for generating random no of requests based on average
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        # limiting no of requests to 15
        if requests > 15:
            requests = 15
        # (0,0) is not considered as customer request, however the driver is free to reject all
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]
        actions = [self.action_space[i] for i in possible_actions_index]
       

        return possible_actions_index, actions

    def new_time_day(self, time, day, ride_duration):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)
        new_time_of_day = time + ride_duration
        new_day_of_week = day
        if new_time_of_day > 23:
            new_time_of_day = new_time_of_day % 24
            new_day_of_week += 1
            if new_day_of_week > 6:
                new_day_of_week = new_day_of_week % 7
        return new_time_of_day,new_day_of_week

  
    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        # Initialize various times
        total_time   = 0
        pickup_time = 0    # time from current  location to pickup location
        waiting_time    = 0    # time if driver to refuse all requests
        drop_time    = 0    # time from  Pick-up point to drop point
        
        # getting the current location, time, day and request locations
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        # 1. driver refuse to requests
        # so wait time is 1 unit, next location is current location
        if ((pickup_loc== 0) and (drop_loc == 0)):
            waiting_time = 1
            next_loc = curr_loc
            
        # 2. cab is already at pick up point
        #if current cab position is same as  pick up position
        elif (curr_loc == pickup_loc):
            drop_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            
            # next location is the drop location
            next_loc = drop_loc
        # 3. cab is not at the pickup point
        else:
            # Driver is away to pickup point, he has to travel to pickup point first
            # time take to reach pickup point
            pickup_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.new_time_day(curr_time, curr_day, pickup_time)
            
            # we calculated pickup Time, now time taken to drop
            drop_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (waiting_time + pickup_time + drop_time)
        next_time, next_day = self.new_time_day(curr_time, curr_day, total_time)
        
        # Construct next_state using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state, waiting_time, pickup_time, drop_time
    

    def reset(self):
        """Return the current state and action space"""
        return self.action_space, self.state_space, self.state_init

    def reward_func(self, waiting_time, pickup_time, drop_time):
        """Takes in state, action and Time-matrix and returns the reward"""
        # pickup time and waiting time we will not get revenue,only we loose battery costs, so they are idle times.
        passenger_time = drop_time
        idle_time      = waiting_time + pickup_time
        
        reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        return reward

    def step(self, state, action, Time_matrix):
        """
        get rewards, next step and total time spent
        """
        # Get the next state and the other time durations
        next_state, waiting_time, pickup_time, drop_time = self.next_state_func(
            state, action, Time_matrix)

        # getting the reward for the ride
        rewards = self.reward_func(waiting_time, pickup_time, drop_time)
        total_time = waiting_time + pickup_time + drop_time
        
        return rewards, next_state, total_time