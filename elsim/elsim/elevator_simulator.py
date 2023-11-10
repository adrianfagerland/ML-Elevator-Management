import csv
import time
from datetime import datetime
from random import Random
from typing import Callable

import numpy as np
from numpy import exp
from numpy import infty as INFTY

from elsim.elevator import Elevator


class ElevatorSimulator:
    """Class for running an decision algorithm as a controller for a simulated elevator system in a building.

    """

    def __init__(self,
                 num_floors: int,
                 num_elevators: int,
                 speed_elevator: float = 2.0,
                 acceleration_elevator: float = 0.4,
                 max_load: int = 7,
                 counter_weight: float = 0.4,
                 random_init: bool = False,
                 random_seed: float = 0):
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

        # Store simulation parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_speed = speed_elevator
        self.max_acc = acceleration_elevator
        self.max_load = max_load
        self.random_init = random_init  # currently ignored and 0 is used :TODO
        self.done = False

        self.r = Random(random_seed)

        # Init Elevators
        self.elevators = [Elevator(
            0, self.num_floors, self.max_speed, self.max_acc) for _ in range(self.num_elevators)]

        # People positioning
        self.floor_queue_list_up = [list() for _ in range(self.num_floors)]
        self.floor_queue_list_down = [list() for _ in range(self.num_floors)]

        # Each elevator has a list in which every current passanger is represented by a tuple
        # each tuple consists of (arriving time, entry elevator time, target floor)
        self.elevator_riding_list: list[list[tuple[float, float, int]]] = [
            list() for _ in range(self.num_elevators)]
        self.elevator_buttons_list = [
            [0 for _ in range(self.num_floors)] for _ in range(self.num_elevators)]

        # TODO update buttons press to link to waiting queue length
        # up and down buttons on each floor
        self.floor_buttons_pressed_up = [0 for _ in range(self.num_floors)]
        self.floor_buttons_pressed_down = [0 for _ in range(self.num_floors)]

        # loss parameters
        self.decay_rate = 0.02  # 1minute ^= 30%

    def read_in_people_data(self, path: str):
        """ Reads the csv file of the arrivals. Stores the arrivals in self.arrivals.

        Args:
            path (str): path to the csv file
        """
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            all_arrivals = [[datetime.fromisoformat(row[0]), int(row[1]), int(row[2])] for row in reader]

        # Check that all specified floors are valid in this building
        assert min([arrivals[2] for arrivals in all_arrivals]) >= 0 and max(
            [arrivals[2] for arrivals in all_arrivals]) < self.num_floors
        assert min([arrivals[1] for arrivals in all_arrivals]) >= 0 and max(
            [arrivals[1] for arrivals in all_arrivals]) < self.num_floors

        start_time = all_arrivals[0][0]
        self.arrivals = [((arrival[0] - start_time).total_seconds(), arrival[1], arrival[2])
                         for arrival in all_arrivals]

    def init_simulation(self, path: str):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.

        Args:
            path (str): path to the csv file containing the arrival data with the specified format.
        """
        self.read_in_people_data(path)

        # start clock for simulation
        self.world_time = 0
        self.next_arrival_index = 0

    def return_observations(self, step_size):
        elevator_doors = np.array([elevator.get_doors_open() for elevator in self.elevators])
        elevator_positions_speed = np.array([(elevator.get_position(), elevator.get_speed())
                                            for elevator in self.elevators])
        elevator_buttons = np.array(self.elevator_buttons_list)
        elevator_target = np.array([elevator.get_target_position() for elevator in self.elevators])

        floor_buttons = np.array(list(zip(self.floor_buttons_pressed_up, self.floor_buttons_pressed_down)))

        loss = self.loss_calculation(step_size)

        # done = False  # TODO write code that determine if done

        observations = {
            "position": elevator_positions_speed[:, 0],
            "speed": elevator_positions_speed[:, 1],
            "doors_state": elevator_doors,
            "buttons": elevator_buttons,
            "target": elevator_target,
            "floors": floor_buttons,
        }
        return (observations, -loss, self.done, None)

    def reset_simulation(self):
        """ Resets the simulation by bringing simulation back into starting state
        """
        # TODO
        self.done = False
        return self.return_observations(step_size=0)

    def loss_calculation(self, time_step: float) -> float:
        """ Calculates the loss afte calling the step() function for the current step()

        Args:
            time_step (float): [the time the last step took in seconds]

        Returns:
            float: [the complete loss scaled down for a reasonable size]
        """
        total_loss = 0

        # loop over all person and add their ind_loss to total loss
        for riding_list in self.elevator_riding_list:
            for person_riding in riding_list:
                # get individual loss
                total_loss += self._ind_loss(time_step, person_riding[0])

        for waiting_queue in self.floor_queue_list_down:
            for waiting_person in waiting_queue:
                total_loss += self._ind_loss(time_step, waiting_person[0])

        # also punish elevator movement
        return total_loss

    def _ind_loss(self, time_step: float, x_0: float) -> float:
        """ Calculates the loss that an indiviual person contributes to the total loss.

        Args:
            time_step (float): the time the person had to wait for which to calculate the loss
            x_0 (float): the time length the person had to wait before the current step

        Returns:
            float: the loss for that person
        """
        ind_loss = (self.decay_rate**2*x_0**2+2*self.decay_rate*x_0+2)/self.decay_rate**3 -  \
                   (exp(-self.decay_rate*time_step) * (self.decay_rate**2*x_0**2+(2*self.decay_rate**2*time_step+2*self.decay_rate)*x_0 +
                                                       self.decay_rate**2*time_step**2+2*self.decay_rate*time_step+2))/self.decay_rate**3

        return ind_loss

    def step(self, actions) -> dict:

        # TODO: Execute actions
        targets = actions['target']
        next_movements = actions['to_serve']
        for i, elevator in enumerate(self.elevators):
            elevator.set_target_position(targets[i], next_movements[i])

        # find out when next event happens that needs to be handled by decision_algorithm
        # => either an elevator arrives or a person arrives

        # Get next person arrival if no person left set time to arrival to infty
        if (self.next_arrival_index >= len(self.arrivals)):
            next_arrival, floor_start, floor_end = INFTY, 0, 0
        else:
            next_arrival, floor_start, floor_end = self.arrivals[self.next_arrival_index]

        # Get next elevator arrival
        elevator_arrival_times = [(ind, elevator.get_time_to_target())
                                  for ind, elevator in enumerate(self.elevators)]
        next_elevator_index, next_elevator_time = sorted(elevator_arrival_times, key=lambda x: x[1])[0]

        # Check if no person left to transport and if no elevator still on its way to a target floor then exit simulation
        if (min(next_arrival, next_elevator_time) >= INFTY):
            # break
            # TODO: discuss how to handle run out of data in the context of learning
            self.done = True
            # raise NotImplementedError

        if (next_arrival < self.world_time + next_elevator_time):
            # update the time of the simulation and remember how big the interval was (for the loss function)
            step_size = next_arrival - self.world_time
            self.world_time = next_arrival
            # simulate elevators till person arrives
            for elevator in self.elevators:
                elevator.advance_simulation(next_arrival - self.world_time)

            # person arrives. Add them to the right queues and update the buttons pressed
            if (floor_end > floor_start):
                self.floor_queue_list_up[floor_start].append(
                    (next_arrival, floor_end))
                self.floor_buttons_pressed_up[floor_start] = 1
            elif (floor_end < floor_start):
                self.floor_queue_list_down[floor_start].append(
                    (next_arrival, floor_end))
                self.floor_buttons_pressed_down[floor_start] = 1
            else:
                raise Exception(
                    "Wrong person input: Target Floor and Start Floor are equal")
            self.next_arrival_index += 1

        else:
            # update the time of the simulation and remember how big the interval was (for the loss function)
            step_size = next_arrival - self.world_time
            self.world_time = next_arrival

            # simulate elevators till elevator arrives
            for elevator in self.elevators:
                elevator.advance_simulation(next_elevator_time)

            arrived_elevator = self.elevators[next_elevator_index]
            arrived_floor = int(arrived_elevator.trajectory_list[0].position)

            # 1. do people want to leave?
            self.elevator_riding_list[next_elevator_index] = list(filter(lambda x: x[2] == arrived_floor,
                                                                         self.elevator_riding_list[next_elevator_index]))

            # depending on the direction of the elevator: update floors buttons by disabling them
            if (arrived_elevator.next_movement == 1):
                self.floor_buttons_pressed_up[arrived_floor] = 0
                elevator_join_list = self.floor_queue_list_up[arrived_floor]

            elif (arrived_elevator.next_movement == -1):
                self.floor_buttons_pressed_down[arrived_floor] = 0
                elevator_join_list = self.floor_queue_list_down[arrived_floor]
            else:
                # if next direction hasnt not yet been determined
                # if both up and down are pressed but elevator has not given direction (what to do???) currently just going down
                if (len(self.floor_queue_list_down[arrived_floor]) > 0 and len(self.floor_queue_list_up[arrived_floor])):
                    self.floor_buttons_pressed_down[arrived_floor] = 0
                    elevator_join_list = self.floor_queue_list_down[arrived_floor]
                elif (len(self.floor_queue_list_up[arrived_floor]) > 0):
                    self.floor_buttons_pressed_up[arrived_floor] = 0
                    elevator_join_list = self.floor_queue_list_up[arrived_floor]
                elif (len(self.floor_queue_list_down[arrived_floor]) > 0):
                    self.floor_buttons_pressed_down[arrived_floor] = 0
                    elevator_join_list = self.floor_queue_list_down[arrived_floor]
                else:
                    elevator_join_list = []

            # add the people queing on that floor the the elevator riding list if still enough space
            num_possible_join = self.max_load - \
                len(self.elevator_riding_list[next_elevator_index])

            for i in range(min(len(elevator_join_list), num_possible_join)):
                self.elevator_riding_list[next_elevator_index].append((
                    elevator_join_list[i][0], self.world_time, elevator_join_list[i][1]))

            elevator_join_list = elevator_join_list[max(
                len(elevator_join_list), num_possible_join):]

            if (len(elevator_join_list) > 0):
                # not all people could join, press elevator button again after few seconds
                button_press_again_time = self.r.randint(4, 8)
                new_arrival_time = self.world_time + next_elevator_time + button_press_again_time

                # find spot to insert new arrival
                i = self.next_arrival_index
                while (i < len(self.arrivals) and self.arrivals[i][0] < new_arrival_time):
                    i += 1
                for start_time, end_floor in elevator_join_list:
                    self.arrivals.insert(i, (start_time, arrived_floor, end_floor))

            pass

            # update buttons in elevator
            elevator_target_list = [x[1] for x in self.elevator_riding_list[next_elevator_index]]
            self.elevator_buttons_list[next_elevator_index] = [1 if i in elevator_target_list else 0
                                                               for i in range(0, self.num_floors)]

        # Arrivals handled

        # return the data for the observations
        return self.return_observations(step_size=step_size)


if __name__ == "__main__":
    e = ElevatorSimulator(10, 4)
    e.init_simulation("../pxsim/data/test_data.csv")
