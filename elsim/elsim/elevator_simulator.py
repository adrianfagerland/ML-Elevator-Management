
from turtle import pos
from typing import Callable
from elsim.elevator import Elevator
from random import Random
from elsim.parameters import INFTY, DOOR_OPENING_TIME



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

        # Store energy parameters
        self.counter_weight = counter_weight

    
    def init_simulation(self):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.
        """

    def run(self, 
            time_to_run : int,
            decision_algorithm : Callable):
        pass
  
    