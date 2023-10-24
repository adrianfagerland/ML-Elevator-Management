from typing import Callable
from random import Random
from math import sqrt

from torch import sgn
# All time values in seconds
DOOR_OPENING_TIME = 1

DIST_EPSILON = 0.01 # 1/100th of an floor allowed error due to math inaccuracys

INFTY = 2**40 # just a large number

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
        self.elevators = [self.Elevator(0, self.num_floors, self.max_speed, self.max_acc) for _ in range(self.num_elevators)]

        # Store energy parameters
        self.counter_weight = counter_weight

    
    def init_simulation(self):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.
        """

    def run(self, 
            time_to_run : int,
            decision_algorithm : Callable):
        pass
        
    class Elevator:
        """Class for keeping track of an elevator."""
        def __init__(self, 
                     position: float, 
                     num_floors: int, 
                     max_speed: float, 
                     max_acceleration: float, 
                     num_passangers: int = 0,
                     current_speed: float = 0):
            self.current_position = position
            self.num_floors = num_floors
            self.max_speed = max_speed
            self.current_speed = current_speed
            self.max_acceleration = max_acceleration
            self.num_passangers = num_passangers
            self.target_position : int = int(self.current_position)
            self.inner_buttons_pressed: list[bool] =  [False] * num_floors
            self.door_opened_percentage: float = 0
            self.energy_consumption: float = 0

            self.trajectory_list : list[tuple[float, float, float]] = [(self.current_position, self.current_speed, 0)]
            self.time_to_target : float = INFTY

        def set_target_position(self, new_target_position : int):
            self.target_position = new_target_position
            
            # Compute trajectory (lots of maths) :@
            self.update_trajectory()


        def update_trajectory(self):
            self.trajectory_list = self.trajectory_list[0:1]

            # while not at correct position and velocity is 0, get new step in trajectory
            while(not (self.trajectory_list[-1][0] == self.target_position and self.trajectory_list[-1][1] == 0)):
                current_pos = self.trajectory_list[-1][0]
                current_speed = self.trajectory_list[-1][1]
                current_time = self.trajectory_list[-1][2]

                dist = self.target_position - current_pos
                
                if(dist * current_speed < 0):
                    # first completely slow down, before reversing
                    time_to_slow_down = abs(current_speed / self.max_acceleration)
                    acc_for_step = -self.max_acceleration if self.current_speed > 0 else self.max_acceleration

                    new_pos = current_pos + current_speed * time_to_slow_down + 0.5 * acc_for_step *time_to_slow_down**2
                    self.trajectory_list.append((new_pos, 0, time_to_slow_down))
                else:
                    # speed is going in the right direction or is 0

                    # factor determining if we want to reach -max_speed (target is lower than pos) or +max_speed if (target is higher than pos)
                    sgn_factor = -1 if dist < 0 else 1

                    # Calculate the distance traveled while accelerating to max_speed
                    time_to_max_speed = abs(self.max_speed - current_speed) / self.max_acceleration
                    distance_accelerating =  time_to_max_speed * current_speed + 0.5 * sgn_factor * self.max_acceleration * time_to_max_speed**2

                    # Calculate the distance traveled while deaccelerating to 0
                    time_to_slow_down = self.max_speed / self.max_acceleration
                    distance_breaking = sgn_factor * self.max_speed * time_to_slow_down - 0.5 * sgn_factor * self.max_acceleration * time_to_slow_down**2

                    # check if elevator has enough distance to reach max_speed
                    acceleration_distance = distance_accelerating + distance_breaking
                    if(abs(acceleration_distance) <= abs(dist)):
                        # if yes how long with max speed?
                        distance_with_max_speed = abs(dist) - abs(acceleration_distance)
                        time_with_max_speed = distance_with_max_speed / self.max_speed

                        trajetory_step1 = (current_pos + distance_accelerating, sgn_factor * self.max_speed, time_to_max_speed)
                        trajetory_step2 = (current_pos + distance_accelerating + distance_with_max_speed, sgn_factor * self.max_speed, time_with_max_speed)
                        trajetory_step3 = (current_pos + acceleration_distance + distance_with_max_speed, 0, time_to_slow_down)

                        self.trajectory_list.extend([trajetory_step1, trajetory_step2, trajetory_step3])
                    else:
                        # check if going to overshoot the target?
                        time_to_slow_down = abs(current_speed / self.max_acceleration)
                        distance_slow_down = current_speed * time_to_slow_down - 0.5 * sgn_factor * self.max_acceleration * time_to_slow_down**2
                        
                        if(abs(distance_slow_down) > abs(dist)):
                            # is going to overshoot, stop complety after the target and turn around (next iteration of while)
                            self.trajectory_list.append((current_pos + distance_slow_down, 
                                                         current_speed - sgn_factor * self.max_acceleration * time_to_slow_down, 
                                                         time_to_slow_down))
                
                        else:
                            # only other option: not enough time to speed up to max_speed 
                            # annoying math and quite error prone (yikes)
                            # then calculate best trajectory

                            rem_distance = dist - distance_slow_down
                            # PQ formel to solve equation
                            p = 4 * current_speed / self.max_acceleration
                            q = 4 * -abs(rem_distance) / self.max_acceleration

                            time_to_speed_up = (-(p/2) + sqrt((p/2)**2 - q) ) / 2


                            final_speed = current_speed + time_to_speed_up * self.max_acceleration * sgn_factor
                            self.trajectory_list.append((current_pos + rem_distance / 2, final_speed, time_to_speed_up))

                            time_to_slow_down += time_to_speed_up

                            self.trajectory_list.append((current_pos + rem_distance + distance_slow_down, 
                                                         final_speed - sgn_factor * time_to_slow_down * self.max_acceleration, 
                                                         time_to_slow_down))
                            pass

            self.time_to_target = sum([trajectory_step[2] for trajectory_step in self.trajectory_list])

        def get_time_to_target(self) -> float:
            """ Outputs the time this elevator needs until it arrives at the required target and plus the time for opening the door

            Returns:
                float: time in seconds

            """
            return self.time_to_target + self.door_opened_percentage * DOOR_OPENING_TIME
