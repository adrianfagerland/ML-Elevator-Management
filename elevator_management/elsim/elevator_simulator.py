import csv
from datetime import datetime
from random import Random

import numpy as np
from elsim.elevator import Elevator
from elsim.parameters import DOOR_OPENING_TIME, DOOR_STAYING_OPEN_TIME, LOSS_FACTOR
from pxsim.generate import generate_arrivals


class ElevatorSimulator:
    """Class for running an decision algorithm as a controller for a simulated elevator system in a building."""

    # TODO illustrate next_move when door is open and moving to another target, aka when someone has been delivered to that floor but noone has pressed a button there

    def __init__(
        self,
        num_floors: int,
        num_elevators: int,
        speed_elevator: float = 2.0,
        acceleration_elevator: float = 0.4,
        max_elevator_occupancy: int = 7,
        random_seed: float = 0,
        random_elevator_init: bool = True,
    ):
        """Initialises the Elevator Simulation.

        Args:
            num_floors (int): the number of floors
            num_elevators (int): the number of elevators. All elevators can access any floor and all elevators have the same charateristics.
            speed_elevator (float, optional): The max speed of an elevator in floors per second. Defaults to 2.0.
            acceleration_elevator (float, optional): The max acceleration of an elevator in floors per second^2. Defaults to 0.4.
            max_elevator_occupancy (int, optional): Max Number of People that can be transported by any elevator. People will not enter any elevator that is currently on its maximum load. Defaults to 7.
            counter_weight (float, optional): Percentage of max load that is used as a counter weight in the simulation. Relevant for Energy Consumption calculation.
            random_elevator_init (bool, optional): Whether the elevators are initialised with a random floor position. CURRENTLY NOT USED!
        """

        # Store simulation parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_speed = speed_elevator
        self.max_acc = acceleration_elevator
        self.max_elevator_occupancy = max_elevator_occupancy
        self.random_init = random_elevator_init
        self.done = False
        self.active_elevators = set()

        self.r = Random(random_seed)

        # Init Elevators
        if self.random_init:
            self.elevators = [
                Elevator(
                    self.r.randint(0, self.num_floors - 1),
                    self.num_floors,
                    self.max_speed,
                    self.max_acc,
                    max_occupancy=max_elevator_occupancy,
                )
                for _ in range(self.num_elevators)
            ]
        else:
            self.elevators = [
                Elevator(
                    0,
                    self.num_floors,
                    self.max_speed,
                    self.max_acc,
                    max_occupancy=max_elevator_occupancy,
                )
                for _ in range(self.num_elevators)
            ]

        # People positioning
        self.floor_queue_list_up = [list() for _ in range(self.num_floors)]
        self.floor_queue_list_down = [list() for _ in range(self.num_floors)]

        # Each elevator has a list in which every current passanger is represented by a tuple
        # each tuple consists of (arriving time, entry elevator time, target floor)
        self.elevator_riding_list: list[list[tuple[float, float, int]]] = [
            list() for _ in range(self.num_elevators)
        ]
        self.elevator_buttons_list = [
            [0 for _ in range(self.num_floors)] for _ in range(self.num_elevators)
        ]

        # TODO update buttons press to link to waiting queue length
        # up and down buttons on each floor
        self.floor_buttons_pressed_up = [0 for _ in range(self.num_floors)]
        self.floor_buttons_pressed_down = [0 for _ in range(self.num_floors)]

        # loss parameters
        self.decay_rate = 0.02  # 1minute ^= 30%
        self.last_observation_call = 0


    def get_floor_buttons_pressed_up(self):
        return [0 if not floor_queue else 1 for floor_queue in self.floor_queue_list_up]

    def get_floor_buttons_pressed_down(self):
        return [
            0 if not floor_queue else 1 for floor_queue in self.floor_queue_list_down
        ]

    def generate_arrivals_data(self):
        """ Generates arrival data for people. Stores the arrivals in self.arrivals.

        Args:
            path (str): path to the csv file
        """
        all_arrivals = list(generate_arrivals(self.num_floors, self.num_elevators, 1, 150))
        # Check that all specified floors are valid in this building
        assert (
            min([arrivals[2] for arrivals in all_arrivals]) >= 0
            and max([arrivals[2] for arrivals in all_arrivals]) < self.num_floors
        )
        assert (
            min([arrivals[1] for arrivals in all_arrivals]) >= 0
            and max([arrivals[1] for arrivals in all_arrivals]) < self.num_floors
        )

        start_time = all_arrivals[0][0]
        self.arrivals = [
            ((arrival[0] - start_time).total_seconds(), arrival[1], arrival[2])
            for arrival in all_arrivals
        ]

    def init_simulation(self, path: str):
        """Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.

        Args:
            path (str): path to the csv file containing the arrival data with the specified format.
        """
        self.generate_arrivals_data()

        # start clock for simulation
        self.world_time = 0
        self.next_arrival_index = 0

    def get_observations(self, needs_decision=True) -> tuple:
        
        time_since_last = self.world_time - self.last_observation_call
        self.last_observation_call = self.world_time


        elevator_data = []
        for elevator in self.elevators:
            elevator_data.append({
                "position":np.array([elevator.get_position()], dtype=np.float32),
                "speed": np.array([elevator.get_speed()], dtype=np.float32),
                "target": np.array(elevator.get_target_position()),
                "buttons": np.array(elevator.get_buttons()), 
                "doors_state": np.array([elevator.get_doors_open()], dtype=np.float32),
                "doors_moving_direction": np.array([elevator.get_doors_moving_direction()], dtype=np.float32),
            })

        floor_buttons = np.array(
            list(
                zip(
                    self.get_floor_buttons_pressed_up(),
                    self.get_floor_buttons_pressed_down(),
                )
            ),
            dtype=np.int8,
        )

        loss = self.loss_calculation(time_since_last)

        # a dictionary for info, should be used to pass information about the run from
        # then env to an algorithm for logging
        info_dictionary = {'needs_decision':needs_decision}

        time_dictionary = {"time_seconds": np.array([ self.world_time], dtype=np.float32),
                            "time_since_last_seconds":np.array([time_since_last], dtype=np.float32)}

        # create dictionary with corrects types expected from gymnasium
        observations = {
            "floors": floor_buttons,
            "num_elevators": np.array([self.num_elevators], dtype=np.uint8),
            "time": time_dictionary,
            "elevators": tuple(elevator_data)
        }
        #       observation   reward  terminated? truncated? info
        return (observations, -loss, self.done, False, info_dictionary)

    def reset_simulation(self):
        """Resets the simulation by bringing simulation back into starting state"""
        # TODO
        self.last_observation_call = 0
        self.done = False
        obs, _, _, _, info = self.get_observations()
        return obs, info

    def loss_calculation(self, time_step) -> float:
        """Calculates the loss afte calling the step() function for the current step()

        Args:
            time_step (float): [the time the last step took in seconds]

        Returns:
            float: [the complete loss scaled down for a reasonable size]
        """
        total_loss = 0

        # loop over all person and add their ind_loss to total loss
        for elevator in self.elevators:
            for rider in elevator.get_riders():
                # get individual loss
                total_loss += self._ind_loss2(time_step, rider[0]) / 5

        for waiting_queue in self.floor_queue_list_down:
            for waiting_person in waiting_queue:
                total_loss += self._ind_loss2(time_step, waiting_person[0])

        # also punish elevator movement
        return total_loss / LOSS_FACTOR

    def _ind_loss(self, time_step: float, x_0: float) -> float:
        """Calculates the loss that an indiviual person contributes to the total loss.

        Args:
            time_step (float): the time the person had to wait for which to calculate the loss
            x_0 (float): the time length the person had to wait before the current step

        Returns:
            float: the loss for that person
        """
        ind_loss = (
            self.decay_rate**2 * x_0**2 + 2 * self.decay_rate * x_0 + 2
        ) / self.decay_rate**3 - (
            np.exp(-self.decay_rate * time_step)
            * (
                self.decay_rate**2 * x_0**2
                + (2 * self.decay_rate**2 * time_step + 2 * self.decay_rate) * x_0
                + self.decay_rate**2 * time_step**2
                + 2 * self.decay_rate * time_step
                + 2
            )
        ) / self.decay_rate**3

        return ind_loss
    def _ind_loss2(self, time_step: float, x_0: float) -> float:
        """Calculates the loss that an indiviual person contributes to the total loss.

        Args:
            time_step (float): the time the person had to wait for which to calculate the loss
            x_0 (float): the time length the person had to wait before the current step

        Returns:
            float: the loss for that person
        """
        ind_loss = 1/3 * ((self.world_time - x_0)**3 - (self.world_time - time_step - x_0)**3)

        return ind_loss

    def _handle_arrivals_departures(self, next_elevator: Elevator):
        if next_elevator.get_doors_open() != 1:
            return
        arrived_floor = int(next_elevator.get_position())
        if (
            next_elevator.get_target_position() == next_elevator.get_position()
            and next_elevator.next_movement == 0
        ):
            next_elevator.vibe_check(arrived_floor=arrived_floor)
        elif next_elevator.next_movement != 0:
            next_elevator.vibe_check(arrived_floor=arrived_floor)
            if next_elevator.next_movement == 1:
                target_queue = self.floor_queue_list_up
            elif next_elevator.next_movement == -1:
                target_queue = self.floor_queue_list_down
            else:
                raise Exception("Something went wrong")
            num_possible_join = next_elevator.get_num_possible_join()
            for i in range(min(len(target_queue[arrived_floor]), num_possible_join)):
                next_elevator.add_rider(
                    (
                        target_queue[arrived_floor][i][0],
                        self.world_time,
                        target_queue[arrived_floor][i][1],
                    )
                )
            del target_queue[arrived_floor][
                : min(len(target_queue[arrived_floor]), num_possible_join)
            ]

            if len(target_queue[arrived_floor]) > num_possible_join:
                # not all people could join, press elevator button again after few seconds
                button_press_again_time = DOOR_STAYING_OPEN_TIME + DOOR_OPENING_TIME + 1
                new_arrival_time = self.world_time + button_press_again_time

                # find spot to insert new arrival
                i = self.next_arrival_index
                while i < len(self.arrivals) and self.arrivals[i][0] < new_arrival_time:
                    i += 1
                for start_time, end_floor in target_queue[arrived_floor]:
                    self.arrivals.insert(i, (start_time, arrived_floor, end_floor))

                    
    def step(self, actions, max_step_size=None) -> tuple:

        # if action is defined => execute the actions by sending them to the elevators
        if actions is not None:
            targets = actions["target"]
            next_movements = actions["next_move"]
            for i, elevator in enumerate(self.elevators):
                elevator.set_target_position(targets[i], next_movements[i])
                if (
                    elevator.get_target_position() != elevator.get_position()
                    or next_movements[i] != 0
                    and elevator not in self.active_elevators
                ):
                    self.active_elevators.add(elevator)

        # find out when next event happens that needs to be handled by decision_algorithm
        # => either an elevator arrives or a person arrives

        # Get next person arrival if no person left set time to arrival to infty
        if self.next_arrival_index >= len(self.arrivals):
            next_arrival, floor_start, floor_end = np.infty, 0, 0
        else:
            next_arrival, floor_start, floor_end = self.arrivals[
                self.next_arrival_index
            ]

        # Check if no person left to transport and if no elevator still on its way to a target floor then exit simulation
        # if (min(next_arrival, next_elevator_time) >= np.infty):
        if next_arrival >= np.infty:
            # break
            # TODO: discuss how to handle run out of data in the context of learning
            self.done = True
            # raise NotImplementedError

        # Get next elevator arrival
        next_elevator: Elevator | None = None
        elevator_arrival_times = [(elevator, elevator.get_time_to_target()) for elevator in self.active_elevators ]

        if len(elevator_arrival_times) == 0:
            next_elevator_time = np.infty
        else:
            next_elevator, next_elevator_time = min(
                elevator_arrival_times, key=lambda x: x[1]
            )

        # Test if max_step_size is less than the next event, then just advance simulation max_step_size
        if (max_step_size is not None and
            max_step_size < min(next_elevator_time,next_arrival - self.world_time)):

            for elevator in self.elevators:
                elevator.advance_simulation(max_step_size)
                self._handle_arrivals_departures(elevator)
            self.world_time += max_step_size
            return self.get_observations(needs_decision=False)

        step_size = next_arrival - self.world_time
        if next_arrival < self.world_time + next_elevator_time:
            # update the time of the simulation and remember how big the interval was (for the loss function)
            self.world_time = next_arrival
            # simulate elevators till person arrives
            for elevator in self.elevators:
                elevator.advance_simulation(next_arrival - self.world_time)

            # person arrives. Add them to the right queues and update the buttons pressed
            if floor_end > floor_start:
                self.floor_queue_list_up[floor_start].append((next_arrival, floor_end))
            elif floor_end < floor_start:
                self.floor_queue_list_down[floor_start].append(
                    (next_arrival, floor_end)
                )
            else:
                raise Exception(
                    "Wrong person input: Target Floor and Start Floor are equal"
                )
            self.next_arrival_index += 1
        
        # had to disable this as this was causing problems, and i (simon) dont see why this was neccessary 
        # in the first place. only commented it out as someone might have had a reason
        """elif next_elevator.next_movement == 0:
            next_elevator.vibe_check(arrived_floor=next_elevator.get_position())
            self.active_elevators.remove(
                next_elevator
            )  # TODO check if this makes sense #doubt
        """ 
        if(next_arrival > self.world_time + next_elevator_time):
            # update the time of the simulation and remember how big the interval was (for the loss function)
            self.world_time += next_elevator_time

            # simulate elevators till elevator arrives
            for elevator in self.elevators:
                elevator.advance_simulation(next_elevator_time)
            
            assert next_elevator is not None
            self._handle_arrivals_departures(next_elevator)
        # Arrivals handled

        # return the data for the observations
        return self.get_observations()

