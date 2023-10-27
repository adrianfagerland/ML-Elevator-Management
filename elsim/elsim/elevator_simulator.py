
import time
from turtle import pos
from typing import Callable
from random import Random
import csv
from datetime import datetime

from elsim.elevator import Elevator
from elsim.parameters import INFTY, DOOR_OPENING_TIME, DATETIME_FORMAT_INPUT

class ElevatorSimulator:
    """Class for running an decision algorithm as a controller for a simulated elevator system in a building.

    """

    def __init__(self, 
                 num_floors : int,
                 num_elevators : int,
                 speed_elevator : float = 2.0,
                 acceleration_elevator : float = 0.4,
                 max_load : int = 7,
                 counter_weight : float = 0.4,
                 random_init : bool = False,
                 random_seed : float = 0):
        """ Initialises the Elevator Simulation.

        Args:
            num_floors (int): the number of floors
            num_elevators (int): the number of elevators. All elevators can access any floor and all elevators have the same charateristics.
            speed_elevator (float, optional): The max speed of an elevator in floors per second. Defaults to 2.0.   
            acceleration_elevator (float, optional): The max acceleration of an elevator in floors per second^2. Defaults to 0.4.
            max_load (int, optional): Max Number of People that can be transported by any elevator. People will not enter any elevator that is currently on its maximum load. Defaults to 7.
            counter_weight (float, optional): Percentage of max load that is used as a counter weight in the simulation. Relevant for Energy Consumption calculation.
            random_init (bool, optional): Whether the elevators are initialised with a random floor position. CURRENTLY NOT USED!
        """
        
        ## Store simulation parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_speed = speed_elevator
        self.max_acc = acceleration_elevator
        self.max_load = max_load
        self.random_init = random_init # currently ignored and 0 is used :TODO

        self.r = Random(random_seed)
        
        # Init Elevators
        self.elevators = [Elevator(0, self.num_floors, self.max_speed, self.max_acc) for _ in range(self.num_elevators)]

        # People positioning
        self.floor_queue_list = [list() for _ in range(self.num_floors)]
        self.elevator_riding_list = [list() for _ in range(self.num_elevators)]

        # up and down buttons on each floor
        self.floor_buttons_pressed_up = [0 for _ in range(self.num_floors)]
        self.floor_buttons_pressed_down = [0 for _ in range(self.num_floors)]


    def read_in_people_data(self, path: str):
        """ Reads the csv file of the arrivals. Stores the arrivals in self.arrivals.

        Args:
            path (str): path to the csv file
        """
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            fields = dict(enumerate(next(reader))) # not needed right now
            # read and convert data
            all_arrivals = list(reader)
            all_arrivals = list(map(lambda x: [datetime.strptime(x[0], DATETIME_FORMAT_INPUT), int(x[1]), int(x[2])], all_arrivals))
            sorted_arrivals : list = sorted(all_arrivals, key=lambda x: x[0])
            
        # convert datetime to seconds since first arrival
        start_time = sorted_arrivals[0][0]
        self.arrivals = list(map(lambda x: ((x[0] - start_time).total_seconds(), x[1],x[2]), sorted_arrivals))

    def init_simulation(self):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.
        """

    def run(self, 
            time_to_run : int,
            decision_algorithm : Callable):
        
        # read data in 
        self.read_in_people_data("..\pxsim\elevator_data.csv")
        
        # start clock for simulation
        world_time = 0

        arrival_index = 0

        # while not running
        while(world_time < time_to_run):
            # get next event that needs to be handled by decision_algorithm 
            # => either an elevator arrives or a person arrives

            next_arrival = self.arrivals[arrival_index][0]

            next_elevator_event = sorted([elevator.get_time_to_target() for elevator in self.elevators])[0]


            if(next_arrival > world_time + next_elevator_event):
                # person arrives

                pass
            else:
                # elevator arrives

                pass

  
    
if __name__ == "__main__":
    e = ElevatorSimulator(10,4)
    e.run(10, lambda x:x)