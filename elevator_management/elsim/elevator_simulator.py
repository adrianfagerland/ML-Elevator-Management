import datetime
import heapq
from collections import defaultdict
from random import Random

import numpy as np
from elsim.elevator import Elevator
from elsim.parameters import (
    DOOR_OPENING_TIME,
    DOOR_STAYING_OPEN_TIME,
    INFTY,
    REWARD_FORGOT_PEOPLE,
    REWARD_JOINING_ELEVATOR,
    REWARD_NORMALIZER,
    Person,
)
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
        num_arrivals: int = 2000,
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

        self.num_arrivals = num_arrivals
        self.r = Random(random_seed)

        self.total_num_arrivals: int = 0

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
        self.floor_queue_list_up: list[list[Person]] = [list() for _ in range(self.num_floors)]
        self.floor_queue_list_down: list[list[Person]] = [list() for _ in range(self.num_floors)]

        # Each elevator has a list in which every current passanger is represented by a tuple
        # each tuple consists of (arriving time, entry elevator time, target floor)
        self.elevator_riding_list: list[list[tuple[float, float, int]]] = [list() for _ in range(self.num_elevators)]
        self.elevator_buttons_list = [[0 for _ in range(self.num_floors)] for _ in range(self.num_elevators)]

        # loss parameters
        self.total_reward: float = 0
        self.recent_reward: float = 0

        # parameters for info return
        self.num_people_used_stairs: int = 0

        self.last_observation_call = 0
        self.world_time: float = 0

    def get_floor_buttons_pressed_up(self):
        return [0 if not floor_queue else 1 for floor_queue in self.floor_queue_list_up]

    def get_floor_buttons_pressed_down(self):
        return [0 if not floor_queue else 1 for floor_queue in self.floor_queue_list_down]

    def generate_arrivals_data(self, density=1):
        """Generates arrival data for people. Stores the arrivals in self.arrivals.

        Args:
            path (str): path to the csv file
        """

        all_arrivals = list(generate_arrivals(self.num_floors, self.num_elevators, density, self.num_arrivals))

        # Check that all specified floors are valid in this building
        assert (
            min([arrivals[2] for arrivals in all_arrivals]) >= 0
            and max([arrivals[2] for arrivals in all_arrivals]) < self.num_floors
        )
        assert (
            min([arrivals[1] for arrivals in all_arrivals]) >= 0
            and max([arrivals[1] for arrivals in all_arrivals]) < self.num_floors
        )

        # assert there are at least 1 arrival
        assert len(all_arrivals) > 0

        # We do not always want the first person to arrive at time 0
        # Therefore, we shift the arrival times so that the first person arrives after the same interval
        # as between the first and second person
        first_arrival = all_arrivals[0][0]
        time_shift = first_arrival - datetime.timedelta(seconds=self.r.randint(1, 10))
        self.original_arrivals: list[Person] = [
            Person(
                (arrival[0] - time_shift).total_seconds(),
                arrival[1],
                arrival[2],
                (arrival[0] - time_shift).total_seconds(),
            )
            for arrival in all_arrivals
        ]  # don't know if we need to keep track of the original arrivals
        self.arrivals = self.original_arrivals.copy()
        heapq.heapify(self.arrivals)

    def init_simulation(self, density):
        """Parameters should be the running time and how many people, i.e. all the information that the arrival generation needs. Also an instance of the control algorithm class.

        Args:
            path (str): path to the csv file containing the arrival data with the specified format.
        """
        self.generate_arrivals_data(density)

        # start clock for simulation
        self.world_time = 0

    def get_observations(self, needs_decision=True) -> tuple:
        time_since_last = self.world_time - self.last_observation_call
        self.last_observation_call = self.world_time

        elevator_data = []
        for elevator in self.elevators:
            elevator_data.append(
                {
                    "position": np.array([elevator.get_position()], dtype=np.float32),
                    "speed": np.array([elevator.get_speed()], dtype=np.float32),
                    "target": np.array(elevator.get_target_position()),
                    "buttons": np.array(elevator.get_buttons()),
                    "doors_state": np.array([elevator.get_doors_open()], dtype=np.float32),
                    "doors_moving_direction": np.array([elevator.get_doors_moving_direction()], dtype=np.float32),
                }
            )

        floor_buttons = np.array(
            list(
                zip(
                    self.get_floor_buttons_pressed_up(),
                    self.get_floor_buttons_pressed_down(),
                )
            ),
            dtype=np.int8,
        )

        reward = self.reward_calculation(time_since_last)

        # a dictionary for info, should be used to pass information about the run from
        # then env to an algorithm for logging
        info_dictionary = {
            "needs_decision": needs_decision,
            "num_walked_stairs": self.num_people_used_stairs,
            "total_reward": self.total_reward,
            "recent_reward": reward,
            "num_people_arrived": self.total_num_arrivals,
            "num_people_left_in_sim": self.get_number_of_people_in_sim(),
            "num_people_yet_to_arrive": len(self.arrivals),
            "total_arrivals": len(self.original_arrivals),
        }

        # if done then add some special info for info_dictionary
        if self.done:
            average_arrival_time = 0
            for person in self.original_arrivals:
                if (
                    person.time_left_simulation is not None
                ):  # previous assertion here changed to facilitate for the tests
                    average_arrival_time += person.time_left_simulation - person.arrival_time

            info_dictionary["average_arrival_time"] = average_arrival_time / len(self.original_arrivals)

        time_dictionary = {
            "time_seconds": np.array([self.world_time], dtype=np.float32),
            "time_since_last_seconds": np.array([time_since_last], dtype=np.float32),
        }

        # create dictionary with corrects types expected from gymnasium
        observations = {
            "floors": floor_buttons,
            "num_elevators": np.array([self.num_elevators], dtype=np.uint8),
            "time": time_dictionary,
            "elevators": tuple(elevator_data),
        }
        #       observation   reward  terminated? truncated? info
        return (observations, reward, self.done, False, info_dictionary)

    def reset_simulation(self):
        """Resets the simulation by bringing simulation back into starting state"""
        # TODO
        self.last_observation_call = 0
        self.done = False
        obs, _, _, _, info = self.get_observations()
        return obs, info

    def reward_calculation(self, time_step) -> float:
        """Calculates the reward afte calling the step() function for the current step()

        Args:
            time_step (float): [the time the last step took in seconds]

        Returns:
            float: [the reward since the last reward call scaled down for a reasonable size]
        """
        reward = self.recent_reward
        self.recent_reward = 0
        return reward

    def add_reward(self, reward_value):
        self.recent_reward += reward_value / REWARD_NORMALIZER
        self.total_reward += reward_value / REWARD_NORMALIZER

    def _handle_arrivals_departures(self, next_elevator: Elevator):
        # only let people leave and join if the doors of the elevator are open
        if next_elevator.get_doors_open() == 1:
            # find people that want to leave
            arrived_floor = int(next_elevator.get_position())
            # people leave on the floor
            left_passengers = next_elevator.passengers_arrive()
            self.total_num_arrivals += len(left_passengers)
            for passenger in left_passengers:
                self.add_reward(passenger.get_arrive_reward(self.world_time))

            # check if people have left floor because wait time too long
            left_queue_up = self.update_wait_queues(self.floor_queue_list_up[arrived_floor])
            left_queue_down = self.update_wait_queues(self.floor_queue_list_down[arrived_floor])

            # keep track of people that used stairs
            for passenger in left_queue_down + left_queue_up:
                self.add_reward(passenger.get_walk_reward(self.world_time))
            self.num_people_used_stairs += len(left_queue_down) + len(left_queue_up)

            target_queue_floor = None
            if next_elevator.get_next_movement() == 1:
                target_queue_floor = self.floor_queue_list_up[arrived_floor]
            elif next_elevator.get_next_movement() == -1:
                target_queue_floor = self.floor_queue_list_down[arrived_floor]

            # if no people are waiting then no one can join
            if target_queue_floor is not None:
                # get possible number of joining people
                num_possible_join = next_elevator.get_num_possible_join()
                # update the waiting queue if people have left

                # take minimum of possible joins and wanted joins
                actual_number_of_joins = min(len(target_queue_floor), num_possible_join)

                people_joining = target_queue_floor[:actual_number_of_joins]
                # and add each one to the elevator
                for joining_person in people_joining:
                    joining_person.entry_elevator_time = self.world_time
                    next_elevator.add_passenger(joining_person)

                    joining_person.entered_elevator(self.world_time)
                    self.add_reward(REWARD_JOINING_ELEVATOR)

                    # remove the person from the target queue
                    target_queue_floor.remove(joining_person)

                # test if all people could join
                if len(target_queue_floor) > 0:
                    # not all people could join, press elevator button again after few seconds
                    button_press_again_time = DOOR_STAYING_OPEN_TIME + DOOR_OPENING_TIME + 3
                    new_arrival_time = self.world_time + button_press_again_time

                    for person in target_queue_floor:
                        # set artificial new arrival time
                        person.arrival_time = new_arrival_time

                        # person cannot have already arrived
                        assert person.time_left_simulation is None
                        # every person that is waiting must have already arrived
                        assert person not in self.arrivals
                        heapq.heappush(self.arrivals, person)

                    # reset the target_queue_floor as the people inserted into arrivals are no longer
                    # waiting (temporarily until they arrive again)
                    target_queue_floor[:] = []

    def get_number_of_people_in_sim(self):
        """
        Finds the number of people that are currently in the simulation. Both waiting on a floor or riding in an elevator.
        """
        waiting_up = sum([len(queue_list) for queue_list in self.floor_queue_list_up])
        waiting_down = sum([len(queue_list) for queue_list in self.floor_queue_list_down])

        riding_elevator = sum([len(elevator.get_passengers()) for elevator in self.elevators])
        return waiting_up + waiting_down + riding_elevator

    def step(self, actions, max_step_size=None) -> tuple:
        # if action is defined => execute the actions by sending them to the elevators
        # print a warning from the warning library if max_step_size is higher than the default
        if actions is not None:
            assert len(actions) == self.num_elevators
            for i, elevator in enumerate(self.elevators):
                elevator_decision = actions[i]
                elevator.set_target_position(
                    elevator_decision["target"],
                    elevator_decision["next_move"],
                )

        # find out when next event happens that needs to be handled by decision_algorithm
        # => either an elevator arrives or a person arrives

        # Get next person arrival if no person left set time to arrival to infty
        if len(self.arrivals) == 0:
            next_arrival = INFTY
        else:
            next_arrival = self.arrivals[0].arrival_time

        # Get next elevator arrival
        elevator_arrival_times = [(elevator, elevator.get_time_to_target()) for elevator in self.elevators]
        next_elevator, next_elevator_time = min(elevator_arrival_times, key=lambda x: x[1])

        # check if we should exit the simulation
        if next_arrival == INFTY and self.get_number_of_people_in_sim() == 0:
            self.done = True
            return self.get_observations()
        elif next_arrival == INFTY and next_elevator_time == INFTY:
            # if both arrivals and next_elevator are infty but people are still in the simulation, penalize algorithm
            self.add_reward(REWARD_FORGOT_PEOPLE)
            self.done = True
            return self.get_observations()

        step_size = next_arrival - self.world_time
        # Test if max_step_size is less than the next event, then just advance simulation max_step_size
        if max_step_size is not None and max_step_size < min(next_elevator_time, step_size):
            self.world_time += max_step_size
            for elevator in self.elevators:
                elevator.advance_simulation(max_step_size)
            return self.get_observations(needs_decision=False)

        # next event is a person arrival
        if next_arrival <= self.world_time + next_elevator_time:
            # advance simulation until a person arrives
            for elevator in self.elevators:
                elevator.advance_simulation(step_size)
            self.world_time = next_arrival
            self._handle_people_arriving_at_floor()

        # next event is an elevator arrival
        if next_arrival > self.world_time + next_elevator_time:
            # simulate elevators till elevator arrives
            self.world_time += next_elevator_time
            for elevator in self.elevators:
                elevator.advance_simulation(next_elevator_time)
            self._handle_arrivals_departures(next_elevator)

        # return the data for the observations
        return self.get_observations()

    def update_wait_queues(self, floor_queue: list[Person]) -> list[Person]:
        """
        Removes people from the wait list if they have waited too long. Returns list of removed people
        """

        still_waitingq_ueue = [person for person in floor_queue if not person.used_stairs(self.world_time)]
        left_queue = [person for person in floor_queue if person.used_stairs(self.world_time)]
        floor_queue[:] = still_waitingq_ueue
        return left_queue

    def _handle_people_arriving_at_floor(self):
        # it has to handle when several people arrive at the same time
        while self.arrivals[0].arrival_time == self.world_time:
            arriving_person = heapq.heappop(self.arrivals)
            floor_start = arriving_person.arrival_floor
            floor_end = arriving_person.target_floor

            if floor_end > floor_start:
                self.floor_queue_list_up[floor_start].append(arriving_person)
            elif floor_end < floor_start:
                self.floor_queue_list_down[floor_start].append(arriving_person)
            else:
                raise Exception("Wrong person input: Target Floor and Start Floor are equal")
            if not self.arrivals:
                break
