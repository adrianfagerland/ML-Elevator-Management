from typing_extensions import Self
from dataclasses import dataclass
from random import Random
from math import sqrt
from copy import deepcopy

from elsim.parameters import INFTY, DOOR_OPENING_TIME

class Elevator:
    """Class for keeping track of an elevator."""
    @dataclass
    class Trajectory:
        """ A single step of a trajectory consisting of important information, for the state of an elevator. Needed so one can extrapolate more easily between pairs of trajectories steps. 
        Used for calculating the complete trajectories and how to move along it.
        """
        position: float
        speed: float
        time: float
        
        def copy(self) -> Self:
            return deepcopy(self)
        
        def set_time(self, time: float) -> Self:
            self.time = time
            return self
        def set_position(self, position: float) -> Self:
            self.position = position
            return self
        def set_speed(self, speed: float) -> Self:
            self.speed = speed
            return self
        def get_values(self) -> tuple[float, float, float]:
            return (self.position, self.speed, self.time)
        def __repr__(self) -> str:
            return str((self.position, self.speed, self.time))

    def __init__(self, 
                    current_position: float, 
                    num_floors: int, 
                    max_speed: float, 
                    max_acceleration: float, 
                    num_passangers: int = 0,
                    current_speed: float = 0):
        self.current_position = current_position
        self.num_floors = num_floors
        self.max_speed = max_speed
        self.current_speed = current_speed
        self.max_acceleration = max_acceleration
        self.num_passangers = num_passangers
        self.target_position : int = int(self.current_position)
        self.inner_buttons_pressed: list[bool] =  [False] * num_floors


        self.door_opened_percentage: float = 0
        self.door_is_opening: bool = False
        self.energy_consumption: float = 0

        self.trajectory_list = [self.Trajectory(current_position, current_speed, 0)]
        self._time_target : float = INFTY


    def set_target_position(self, new_target_position : int):
        """ Set the next target position. Can be done while the elevator is moving (i.e., following a trajectorie). 
        Is not going to affect anything if the doors are currently opening as the doors will continue with their plan 
        and will ask for a new target if the doors are fully openend.

        Args:
            new_target_position (int): The floor number.

        Raises:
            Exception: If floor number is not valid.
        """
        if(self.num_floors <= new_target_position or new_target_position < 0):
            raise Exception(f"New Target Floor {new_target_position} is not in the right range of 0 to {self.num_floors}")

        self.target_position = new_target_position
        
        # if the door is currently opening, do not update the trajectory i.e. close the door and move again
        # rather open the door fully, people can then enter or exit (if there are people there)
        # and then set_target_position is called anyway, because a new target is needed
        if(self.are_doors_opening()):
            self._time_target = sum([trajectory_step.time for trajectory_step in self.trajectory_list])
        else:
            self.update_trajectory()
            
    def are_doors_opening(self) -> bool:
        return 1 > self.door_opened_percentage > 0 and self.door_is_opening

    def are_doors_closing(self) -> bool:
        return 1 > self.door_opened_percentage > 0 and not self.door_is_opening

    def update_trajectory(self):
        """ Updates the trajectory if a new target has been set.
        """
        # Compute trajectory (lots of maths) :@
        self.trajectory_list = self.trajectory_list[0:1]
        
        # check if door needs to be handled, add time to close before starting to move

        if(self.door_opened_percentage > 0):
            trajetory_step = self.trajectory_list[0].copy().set_time(self.door_opened_percentage * DOOR_OPENING_TIME)
            self.trajectory_list.append(trajetory_step)

        # while not at correct position and velocity is 0, get new step in trajectory
        while(not (self.trajectory_list[-1].position == self.target_position and self.trajectory_list[-1].speed == 0)):
            current_pos = self.trajectory_list[-1].position
            current_speed = self.trajectory_list[-1].speed

            dist = self.target_position - current_pos
            
            if(dist * current_speed < 0):
                # first completely slow down, before reversing
                time_to_slow_down = abs(current_speed / self.max_acceleration)
                acc_for_step = -self.max_acceleration if self.current_speed > 0 else self.max_acceleration

                new_pos = current_pos + current_speed * time_to_slow_down + 0.5 * acc_for_step *time_to_slow_down**2
                self.trajectory_list.append(self.Trajectory(new_pos, 0, time_to_slow_down))
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

                    trajetory_step1 = self.Trajectory(current_pos + distance_accelerating, sgn_factor * self.max_speed, time_to_max_speed)
                    trajetory_step2 = self.Trajectory(current_pos + distance_accelerating + sgn_factor * distance_with_max_speed, sgn_factor * self.max_speed, time_with_max_speed)
                    trajetory_step3 = self.Trajectory(current_pos + acceleration_distance + sgn_factor * distance_with_max_speed, 0, time_to_slow_down)

                    self.trajectory_list.extend([trajetory_step1, trajetory_step2, trajetory_step3])
                else:
                    # check if going to overshoot the target?
                    time_to_slow_down = abs(current_speed / self.max_acceleration)
                    distance_slow_down = current_speed * time_to_slow_down - 0.5 * sgn_factor * self.max_acceleration * time_to_slow_down**2
                    
                    if(abs(distance_slow_down) > abs(dist)):
                        # is going to overshoot, stop complety after the target and turn around (next iteration of while)
                        self.trajectory_list.append(self.Trajectory(current_pos + distance_slow_down, 
                                                                    current_speed - sgn_factor * self.max_acceleration * time_to_slow_down, 
                                                                    time_to_slow_down))
            
                    else:
                        # only other option: not enough time to speed up to max_speed 
                        # annoying math and quite error prone (yikes)
                        # then calculate best trajectory

                        rem_distance = dist - distance_slow_down
                        # PQ formel to solve equation
                        p = 4 * abs(current_speed) / self.max_acceleration
                        q = 4 * -abs(rem_distance) / self.max_acceleration

                        time_to_speed_up = (-(p/2) + sqrt( (p/2)**2 - q) ) / 2


                        final_speed = current_speed + time_to_speed_up * self.max_acceleration * sgn_factor
                        self.trajectory_list.append(self.Trajectory(current_pos + rem_distance / 2, final_speed, time_to_speed_up))

                        time_to_slow_down += time_to_speed_up

                        self.trajectory_list.append(self.Trajectory(current_pos + rem_distance + distance_slow_down, 
                                                                    final_speed - sgn_factor * time_to_slow_down * self.max_acceleration, 
                                                                    time_to_slow_down))
                        pass
        # target position was reached
        # add open doors
        trajectory_step = self.trajectory_list[-1].copy().set_time(DOOR_OPENING_TIME)
        self.trajectory_list.append(trajectory_step)
        

    def get_time_to_target(self) -> float:
        """ Outputs the time this elevator needs until it arrives at the required target plus the time for opening the door

        Returns:
            float: time in seconds

        """
        # if at target position: time to target is infty as this elevator. Makes sense as we only care about
        # elevators that are not yet at their target
        if(len(self.trajectory_list) == 1):
            return INFTY
        self._time_target = sum([trajectory_step.time for trajectory_step in self.trajectory_list])
        return self._time_target

    def advance_simulation(self, time_step: float):
        """ Moves the elevator alongs its calculated trajectory until time step is up or the trajectory is completed.

        Args:
            time_step (float): the time to advance the simulation in seconds.
        """
        # if only one element in trajectory list => elevator at target position. Do not move
        if(len(self.trajectory_list) == 1):
            return

        # Test Edge Case: Door is opening, meaning only have to either fully open the door or partially further open door
        if(self.door_is_opening):
            
            assert time_step <= DOOR_OPENING_TIME * (1 - self.door_opened_percentage)

            self.door_opened_percentage += time_step / DOOR_OPENING_TIME

            if(self.door_opened_percentage == 1):
                self.door_is_opening = False
                self.trajectory_list = self.trajectory_list[0:1]
            else:
                next_trajectory_step = self.trajectory_list[0].copy().set_time((1-self.door_opened_percentage) * DOOR_OPENING_TIME)
                
                self.trajectory_list[0] = next_trajectory_step
            
            # door opening case fully handled            
            return

        # find first step of trajectory with cummulated simulation time more than time_step
        i = 0
        while(i + 1 < len(self.trajectory_list) and time_step - self.trajectory_list[i+1].time >= 0):
            time_step -= self.trajectory_list[i+1].time
            i += 1
        
        # elevator is moved to end of its trajectory, set position to target and open the door 
        # (as every end of a trajectory will result in open doors)
        if(i + 1 == len(self.trajectory_list)):
            self.trajectory_list = [self.trajectory_list[-1].copy().set_time(0)]
            self.door_opened_percentage = 1
            return

        # did the door close before starting to move?
        # maybe update trajectory with opening and closing property
        if(self.trajectory_list[0].position == self.trajectory_list[1].position and self.trajectory_list[0].speed == self.trajectory_list[1].speed):
            # update door value
            self.door_opened_percentage = 0
        
        self.trajectory_list = self.trajectory_list[i:]

        
        # modify partial executed step along trajectory
        last_pos, last_speed, last_time = self.trajectory_list[0].get_values()
        next_pos, next_speed, next_time = self.trajectory_list[1].get_values()
        
        percentage_excecuted = time_step / next_time

        # case distinctions
        if(last_pos == next_pos):
            # door opens or closes
            new_pos = last_pos
            new_speed = last_speed
            # if more trajectory after door movement => door closes
            if(len(self.trajectory_list) > 1):
                self.door_is_opening = False
                self.door_opened_percentage = (DOOR_OPENING_TIME - time_step) / (DOOR_OPENING_TIME)
            else:
                # door is opening
                self.door_is_opening = True
                self.door_opened_percentage = time_step / DOOR_OPENING_TIME
            pass
            new_time = DOOR_OPENING_TIME - time_step
        else:
            if(last_speed == next_speed):
                new_pos = last_pos + (next_pos - last_pos) * percentage_excecuted
                new_speed = last_speed
                # full speed
                pass
            else:
                # accelerating
                new_speed = last_speed + (next_speed - last_speed) * percentage_excecuted
                new_pos = last_pos + last_speed * time_step + (new_speed - last_speed)/2 * time_step

                pass
            pass

        new_time = next_time - time_step
        
        self.trajectory_list[0] = self.Trajectory(new_pos, new_speed, 0)
        self.trajectory_list[1] = self.trajectory_list[1].copy().set_time(new_time)
        pass

