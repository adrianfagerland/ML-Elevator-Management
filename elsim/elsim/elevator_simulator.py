import time
from typing import Callable
from random import Random
import csv
from datetime import datetime
from elsim.parameters import INFTY
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
            next(reader)  # Skip the header row
            all_arrivals = [[datetime.fromisoformat(row[0]), int(row[1]), int(row[2])] for row in reader]

        start_time = all_arrivals[0][0]
        self.arrivals = [( (arrival[0] - start_time).total_seconds(), arrival[1], arrival[2]) for arrival in all_arrivals]

    def init_simulation(self):
        """ Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.
        """

    def run(self,
            path: str,
            decision_algorithm: Callable
            ):

        self.read_in_people_data(path)

        # start clock for simulation
        world_time = 0

        next_arrival_index = 0

        # while not running
        while True:
            # get next event that needs to be handled by decision_algorithm
            # => either an elevator arrives or a person arrives

            # Get next person arrival if no person left set time to arrival to infty
            if (next_arrival_index >= len(self.arrivals)):
                next_arrival, floor_start, floor_end = INFTY, 0, 0
            else:
                next_arrival, floor_start, floor_end = self.arrivals[next_arrival_index]

            next_elevator_index, next_elevator_time = sorted([(ind, elevator.get_time_to_target(
            )) for ind, elevator in enumerate(self.elevators)], key=lambda x: x[1])[0]

            print(next_arrival)
            # Check if no person left to transport and if no elevator still on its way to a target floor then exit simulation
            if (min(next_arrival, next_elevator_time) >= INFTY):
                break

            if (next_arrival < world_time + next_elevator_time):
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
                next_arrival_index += 1
            else:
                # elevator arrives
                arrived_elevator = self.elevators[next_elevator_index]
                arrived_floor = int(
                    arrived_elevator.trajectory_list[0].position)
                # update floors buttons by disabling them
                if (arrived_elevator.continue_up):
                    self.floor_buttons_pressed_up[arrived_floor] = 0
                else:
                    self.floor_buttons_pressed_down[arrived_floor] = 0

                # 1. do people want to leave?
                self.elevator_riding_list[next_elevator_index] = list(filter(lambda x: x[2] == arrived_floor,
                                                                             self.elevator_riding_list[next_elevator_index]))

                # depending on the direction of the elevator: update floors buttons by disabling them
                if (arrived_elevator.continue_up):
                    self.floor_buttons_pressed_up[arrived_floor] = 0
                    elevator_join_list = self.floor_queue_list_up[arrived_floor]

                else:
                    self.floor_buttons_pressed_down[arrived_floor] = 0
                    elevator_join_list = self.floor_queue_list_down[arrived_floor]

                # add the people queing on that floor the the elevator riding list if still enough space
                num_possible_join = self.max_load - \
                    len(self.elevator_riding_list[next_elevator_index])

                self.elevator_riding_list[next_elevator_index] += elevator_join_list[:min(
                    num_possible_join, len(elevator_join_list))]

                elevator_join_list = elevator_join_list[:max(
                    len(elevator_join_list), num_possible_join)]

                if (len(elevator_join_list) > 0):
                    # not all people could join, press elevator button again after few seconds
                    button_press_again_time = self.r.randint(4, 8)
                    new_arrival_time = world_time + next_elevator_time + button_press_again_time

                    # find spot to insert new arrival
                    i = next_arrival_index
                    while (self.arrivals[i][0] < new_arrival_time):
                        i += 1
                    for start_time, end_floor in elevator_join_list:
                        self.arrivals.insert(i, (start_time, arrived_floor, end_floor))

                pass

                # update buttons in elevator
                elevator_target_list = [x[2] for x in self.elevator_riding_list[next_elevator_index]]
                self.elevator_buttons_list[next_elevator_index] = [1 if i in elevator_target_list else 0
                                                                   for i in range(0, self.num_floors)]
                world_time += next_elevator_time
            # Arrivals handled. DONE!

            # Call ECG algorithm and set target of each elevator if not already set target
            decisions = decision_algorithm()

    

    


if __name__ == "__main__":
    e = ElevatorSimulator(10, 4)
    e.run("../pxsim/data/w1_f9_1.0.1.csv", lambda x: x)
