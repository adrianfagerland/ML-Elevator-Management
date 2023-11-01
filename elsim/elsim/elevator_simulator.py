
import time
from typing import Callable
from random import Random
import csv
from datetime import datetime

from elsim.elevator import Elevator

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
        self.floor_queue_list_up = [list() for _ in range(self.num_floors)]
        self.floor_queue_list_down = [list() for _ in range(self.num_floors)]

        # Each elevator has a list in which every current passanger is represented by a tuple 
        # each tuple consists of (arriving time, entry elevator time, target floor)
        self.elevator_riding_list : list[list[tuple[float, float, int] | None]] = [list() for _ in range(self.num_elevators)]
        self.elevator_buttons_list = [[0 for _ in range(self.num_floors)] for _ in range(self.num_elevators)]


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
            all_arrivals = list(map(lambda x: [datetime.fromisoformat(x[0]), int(x[1]), int(x[2])], all_arrivals))
            sorted_arrivals : list = sorted(all_arrivals, key=lambda x: x[0])
            
        # convert datetime to seconds since first arrival
        start_time = sorted_arrivals[0][0]
        self.arrivals = list(map(lambda x: ((x[0] - start_time).total_seconds(), x[1],x[2]), sorted_arrivals))

    def init_simulation(self):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.
        """

    def run(self,
            path: str,
            time_to_run : int,
            decision_algorithm : Callable):
        
        self.read_in_people_data(path)
        
        # start clock for simulation
        world_time = 0

        next_elevator_windex = 0

        # while not running
        while(world_time < time_to_run):
            # get next event that needs to be handled by decision_algorithm 
            # => either an elevator arrives or a person arrives

            next_arrival, floor_start, floor_end = self.arrivals[next_elevator_windex]

            next_elevator_index, next_elevator_time = sorted([(ind, elevator.get_time_to_target()) for ind,elevator in enumerate(self.elevators)], key=lambda x: x[1])[0]


            if(next_arrival > world_time + next_elevator_time):
                # person arrives. Add them to the right queues and update the buttons pressed
                if(floor_end > floor_start):
                    self.floor_queue_list_up[floor_start].append((next_arrival,floor_end))
                    self.floor_buttons_pressed_up[floor_start] = 1
                elif(floor_end < floor_start):
                    self.floor_queue_list_down[floor_start].append((next_arrival,floor_end))
                    self.floor_buttons_pressed_down[floor_start] = 1
                else:
                    raise Exception("Wrong person input: Target Floor and Start Floor are equal")
            else:
                # elevator arrives
                arrived_elevator = self.elevators[next_elevator_index]
                arrived_floor = int(arrived_elevator.trajectory_list[0].position)
                # update floors buttons by disabling them 
                if(arrived_elevator.continue_up):
                    self.floor_buttons_pressed_up[arrived_floor] = 0
                else:
                    self.floor_buttons_pressed_down[arrived_floor] = 0


                # 1. do people want to leave?
                self.elevator_riding_list[next_elevator_index] = list(filter(lambda x: x == None or x[2] == arrived_floor, 
                                                                             self.elevator_riding_list[next_elevator_index]))
                
                
                # 2. do people want to enter?
                # -> can people enter? i.e. max capacity
                
                # update outer buttons



                pass

  
    
if __name__ == "__main__":
    e = ElevatorSimulator(10,4)
    e.run("../pxsim/data/w1_f9_0.1.0.csv", 10, lambda x:x)