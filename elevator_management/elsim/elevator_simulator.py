import datetime
import heapq
from random import Random
from collections import defaultdict

import numpy as np
from elsim.elevator import Elevator
from elsim.parameters import (
    DOOR_OPENING_TIME,
    DOOR_STAYING_OPEN_TIME,
    LOSS_FACTOR,
    Person,
    REWARD_ARRIVE_TARGET,
    REWARD_CUMMULATIVE_TO_TARGET,
    REWARD_JOINING_ELEVATOR,
    WAITING_MAX_TIME
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

        # Loss function data: stores old 
        self.loss_dict: dict[Person, tuple[float, bool]] = {}
        

        self.total_num_arrivals: int = 0
        self.last_total_num_arrivals: int = 0

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
        self.decay_rate = 0.02  # 1minute ^= 30%
        self.last_observation_call = 0

        # information dict
        self.num_people_used_stairs: int = 0



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

        loss = self.loss_calculation(time_since_last)

        # a dictionary for info, should be used to pass information about the run from
        # then env to an algorithm for logging
        info_dictionary = { "needs_decision": needs_decision,
                            "num_walked_stairs": self.num_people_used_stairs,
                            "num_people_arrived": self.total_num_arrivals,
                            "num_people_left_in_sim": self.get_number_of_people_in_sim(),
                            "num_people_yet_to_arrive": len(self.arrivals),
                            "total_arrivals": len(self.original_arrivals)}
        

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
                total_loss += self._ind_loss3(time_step, rider, elevator.get_position())

        for waiting_queue in self.floor_queue_list_down:
            for waiting_person in waiting_queue:
                total_loss += self._ind_loss3(time_step, waiting_person)
        
        # reward for all people arriving at the goal, regardless of their arrival time TODO: improve
        total_loss -= (self.total_num_arrivals - self.last_total_num_arrivals) * REWARD_ARRIVE_TARGET
        self.last_total_num_arrivals = self.total_num_arrivals
        # also punish elevator movement
        return total_loss

    def _ind_loss(self, time_step: float, person_data: Person) -> float:
        """Calculates the loss that an indiviual person contributes to the total loss.

        Args:
            time_step (float): the time the person had to wait for which to calculate the loss
            x_0 (float): the time length the person had to wait before the current step

        Returns:
            float: the loss for that person
        """
        x_0 = person_data.original_arrival_time

        ind_loss = (self.decay_rate**2 * x_0**2 + 2 * self.decay_rate * x_0 + 2) / self.decay_rate**3 - (
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

    def _ind_loss2(self, time_step: float, person_data: Person) -> float:
        """Calculates the loss that an indiviual person contributes to the total loss.

        Args:
            time_step (float): the time the person had to wait for which to calculate the loss
            x_0 (float): the time length the person had to wait before the current step

        Returns:
            float: the loss for that person
        """
        x_0 = person_data.original_arrival_time

        ind_loss = 1 / 3 * ((self.world_time - x_0) ** 3 - (self.world_time - time_step - x_0) ** 3)

        return ind_loss

    def _ind_loss3(self, time_step: float, person: Person, person_position: float | None = None) -> float:
        ind_reward = 0
        person_has_entered = (person_position is not None)

        # check if person is not in dict then add it 
        if person not in self.loss_dict:
            self.loss_dict[person] = (person.arrival, False)
        
        prev_position, prev_has_entered = self.loss_dict[person]
        
        # test if person has entered elevator in the last step => reward if true
        if not prev_has_entered and person_has_entered:
            ind_reward += REWARD_JOINING_ELEVATOR

        if person_position is None:
            person_position = person.arrival

        # 
        #ind_reward += (person_position - prev_position) / (person.target - person.arrival) * REWARD_CUMMULATIVE_TO_TARGET

        return 0#-ind_reward

    def _handle_arrivals_departures(self, next_elevator: Elevator):
        # only let people leave and join if the doors of the elevator are open
        if next_elevator.get_doors_open() == 1:

            # people leave on the floor
            self.total_num_arrivals += next_elevator.passengers_arrive()



            # now find people that want to enter
            arrived_floor = int(next_elevator.get_position())


            # check if people have left floor because wait time too long
            left_queue_up = self.update_wait_queues_too_long_waiting(self.floor_queue_list_up[arrived_floor])
            left_queue_down = self.update_wait_queues_too_long_waiting(self.floor_queue_list_down[arrived_floor])
            
            # keep track of people that used stairs
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
                    next_elevator.add_rider(joining_person)
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

        riding_elevator = sum([len(elevator.get_riders()) for elevator in self.elevators])
        return waiting_up + waiting_down + riding_elevator



    def step(self, actions, max_step_size=None) -> tuple:
        # if action is defined => execute the actions by sending them to the elevators
        # print a warning from the warning library if max_step_size is higher than the default
        if actions is not None:
            assert len(actions) == self.num_elevators
            for i, elevator in enumerate(self.elevators):
                elevator_decision = actions[i]
                elevator.set_target_position(
                    elevator_decision['target'],
                    elevator_decision['next_move'],
                )

        # find out when next event happens that needs to be handled by decision_algorithm
        # => either an elevator arrives or a person arrives

        # Get next person arrival if no person left set time to arrival to infty
        if len(self.arrivals) == 0:
            next_arrival = np.infty
            if self.get_number_of_people_in_sim() == 0:
                self.done = True
                return self.get_observations()
        else:
            next_arrival = self.arrivals[0].arrival_time

        # Get next elevator arrival
        elevator_arrival_times = [(elevator, elevator.get_time_to_target()) for elevator in self.elevators]
        next_elevator, next_elevator_time = min(elevator_arrival_times, key=lambda x: x[1])

        step_size = next_arrival - self.world_time
        # Test if max_step_size is less than the next event, then just advance simulation max_step_size
        if max_step_size is not None and max_step_size < min(next_elevator_time, step_size):
            self.world_time += max_step_size
            for elevator in self.elevators:
                elevator.advance_simulation(max_step_size)

            return self.get_observations(needs_decision=False)

        if next_arrival <= self.world_time + next_elevator_time:
            # update the time of the simulation and remember how big the interval was # TODO (for the loss function)
            # simulate elevators till person arrives
            for elevator in self.elevators:
                elevator.advance_simulation(step_size)
            self.world_time = next_arrival
            # person arrives. Add them to the right queues and update the buttons pressed
            if(len(self.arrivals) == 0):
                print("ERROR")
            self._handle_people_arriving_at_floor()


        if next_arrival > self.world_time + next_elevator_time:
            # simulate elevators till elevator arrives
            self.world_time += next_elevator_time
            for elevator in self.elevators:
                elevator.advance_simulation(next_elevator_time)
            self._handle_arrivals_departures(next_elevator)

        # Arrivals handled

        # return the data for the observations
        return self.get_observations()

    
    def update_wait_queues_too_long_waiting(self, floor_queue: list[Person]) -> list[Person]:
        """
        Removes people from the wait list if they have waited too long. Returns list of removed people
        """
        
        still_waitingq_ueue = [person for person in floor_queue if not person.used_stairs(self.world_time)]
        left_queue =  [person for person in floor_queue if person.used_stairs(self.world_time)]
        floor_queue[:] = still_waitingq_ueue
        return left_queue
                
    def _handle_people_arriving_at_floor(self):
        # it has to handle when several people arrive at the same time
        while self.arrivals[0].arrival_time == self.world_time:
            arriving_person = heapq.heappop(self.arrivals)
            floor_start = arriving_person.arrival
            floor_end = arriving_person.target
            if floor_end > floor_start:
                self.floor_queue_list_up[floor_start].append(arriving_person)
            elif floor_end < floor_start:
                self.floor_queue_list_down[floor_start].append(arriving_person)
            else:
                raise Exception("Wrong person input: Target Floor and Start Floor are equal")
            if not self.arrivals:
                break

