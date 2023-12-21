from copy import deepcopy
from dataclasses import dataclass
from math import sqrt

import numpy as np
from elsim.parameters import DOOR_OPENING_TIME, DOOR_STAYING_OPEN_TIME, INFTY, Person
from numpy import sign
from typing_extensions import Self

DOORS_OPEN = 1
DOORS_CLOSED = 1 - DOORS_OPEN


class Elevator:
    """Class for keeping track of an elevator."""

    @dataclass
    class Trajectory:
        """A single step of a trajectory consisting of important information, for the state of an elevator. Needed so one can extrapolate more easily between pairs of trajectories steps.
        Used for calculating the complete trajectories and how to move along it.
        """

        position: float
        speed: float
        time: float
        doors_open: float = 0
        # if doors_open_direction = 1  => opening
        # if doors_open_direction = 0  => not moving
        # if doors_open_direction = -1 => closing
        doors_open_direction: int = 0
        next_movement: int = 0

        def copy(self) -> Self:
            copy = deepcopy(self)
            if not (copy.doors_open == 1 and self.doors_open_direction == 0):
                copy.next_movement = 0
            return copy

        def set_open(self, open_percentage: float) -> Self:
            self.doors_open = open_percentage
            return self

        def set_time(self, time: float) -> Self:
            self.time = time
            return self

        def set_position(self, position: float) -> Self:
            self.position = position
            return self

        def set_speed(self, speed: float) -> Self:
            self.speed = speed
            return self

        def set_doors_opening_direction(self, dir: int) -> Self:
            self.doors_open_direction = dir
            return self

        def set_next_movement(self, next_movement: int) -> Self:
            self.next_movement = next_movement
            return self

        def get_doors_opening_direction(self) -> int:
            return self.doors_open_direction

        def get_values(self) -> tuple[float, float, float, float]:
            return (self.position, self.speed, self.time, self.doors_open)

        def are_doors_opening(self) -> bool:
            return self.doors_open_direction == 1

        def __repr__(self) -> str:
            return str(
                (
                    self.position,
                    self.speed,
                    self.time,
                    self.doors_open,
                    self.doors_open_direction,
                )
            )

        def __init__(
            self, position: float, speed: float, time: float, doors_open: float = 0, doors_open_direction: int = 0
        ):
            self.position = position
            self.speed = speed
            self.time = time
            self.doors_open = doors_open
            self.doors_open_direction = doors_open_direction

    def __init__(
        self,
        current_position,
        num_floors: int,
        max_speed: float,
        max_acceleration: float,
        max_occupancy: int = 7,
        current_speed: float = 0,
    ):
        self.num_floors = num_floors
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.target_position: int = current_position
        self.max_occupancy = max_occupancy
        self.buttons: list = [0] * num_floors

        self.passengers: list[Person] = []

        self.trajectory_list: list[Elevator.Trajectory] = [self.Trajectory(current_position, current_speed, 0)]

    def get_next_movement(self) -> int:
        return self.trajectory_list[0].next_movement

    def get_doors_moving_direction(self) -> int:
        return self.trajectory_list[0].doors_open_direction

    def get_num_passangers(self) -> int:
        return len(self.passengers)

    def get_passengers(self) -> list[Person]:
        return self.passengers

    def add_passenger(self, passenger: Person):
        assert self.get_doors_open() == 1
        self.passengers.append(passenger)
        self.buttons[passenger.target_floor] = 1
        self.trajectory_list.insert(
            1, self.trajectory_list[0].copy().set_time(DOOR_STAYING_OPEN_TIME / 2).set_next_movement(0)
        )

    def is_at_floor(self) -> bool:
        """
        Returns true if elevator is at at a floor and people can leave the elevator.
        """
        return self.get_position() == int(self.get_position()) and self.get_speed() == 0 and self.get_doors_open() == 1

    def passengers_arrive(self) -> list[Person]:
        """
        Handle the arrival of an elevator on a floor. Only does something if the doors are open.
        Returns a list of People that left on the floor
        """
        # Test if doors are open and therefore people can leave
        if not self.get_doors_open():
            return []

        arrived_floor = int(self.get_position())

        # If people want to leave on that floor, remove them from riding list.
        new_passengers = [passenger for passenger in self.passengers if passenger.target_floor != arrived_floor]
        left_passengers = [passenger for passenger in self.passengers if passenger.target_floor == arrived_floor]
        self.passengers = new_passengers

        # Served the floor, set button to not pressed
        self.buttons[arrived_floor] = 0
        return left_passengers

    def get_num_possible_join(self):
        return self.max_occupancy - self.get_num_passangers()

    def get_buttons(self):
        return self.buttons

    def set_target_position(self, new_target_position: int, next_movement: int = 0):
        """Set the next target position. Can be done while the elevator is moving (i.e., following a trajectorie).
        Is not going to affect anything if the doors are currently opening as the doors will continue with their plan
        and will ask for a new target if the doors are fully openend.

        Args:
            new_target_position (int): The floor number.
            next_movement (int): Whether the elevator, will continue up (1), down (-1) or not decided yet (0), after arriving at target.
                                 Used to signal waiting passengers, if they should board. Can be violated in the next
                                 set_target_position()

        Raises:
            Exception: If floor number is not valid.
        """
        if isinstance(new_target_position, np.ndarray):
            new_target_position = new_target_position[0]
        if self.num_floors <= new_target_position or new_target_position < 0:
            raise Exception(
                f"New Target Floor {new_target_position} is not in the right range of 0 to {self.num_floors}"
            )
        if (
            self.target_position != new_target_position
            or (self.trajectory_list[0].next_movement != next_movement and self.target_position != self.get_position())
            or (self.trajectory_list[-1].next_movement != next_movement)
            or (  # elevator has just become idle
                self.trajectory_list[0].next_movement == 0
                and next_movement == 0
                and self.get_doors_open() == 1
                and len(self.trajectory_list) == 1
            )
        ):
            self.update_trajectory(new_target_position, next_movement)

    def update_trajectory(self, new_target_position, next_movement):
        """Updates the trajectory if a new target has been set."""
        # Compute trajectory (lots of maths) :@
        self.trajectory_list = self.trajectory_list[0 : 2 if self.is_waiting_for_people() else 1]

        # check if door needs to be handled, add time to close before starting to move
        if (
            self.get_doors_open() != DOORS_CLOSED
            and (
                (self.target_position != self.get_position())
                or (self.trajectory_list[0].next_movement == 0 and not self.is_waiting_for_people())
                or self.target_position != new_target_position
            )
            and not self.are_doors_opening()
        ):
            self.trajectory_list.append(
                self.trajectory_list[0]
                .copy()
                .set_open(DOORS_CLOSED)
                .set_doors_opening_direction(-1)
                .set_time(self.get_doors_open() * DOOR_OPENING_TIME)
                .set_next_movement(0)
            )

        # while not at correct position and velocity is 0, get new step in trajectory
        while not (self.trajectory_list[-1].position == new_target_position and self.trajectory_list[-1].speed == 0):
            current_pos = self.trajectory_list[-1].position
            current_speed = self.trajectory_list[-1].speed

            dist = new_target_position - current_pos

            if dist * current_speed < 0:
                # first completely slow down, before reversing
                time_to_slow_down = abs(current_speed / self.max_acceleration)
                acc_for_step = -self.max_acceleration if current_speed > 0 else self.max_acceleration

                new_pos = current_pos + current_speed * time_to_slow_down + 0.5 * acc_for_step * time_to_slow_down**2
                self.trajectory_list.append(self.Trajectory(new_pos, 0, time_to_slow_down))
            else:
                # speed is going in the right direction or is 0

                # factor determining if we want to reach -max_speed (target is lower than pos) or +max_speed if (target is higher than pos)
                sgn_factor = -1 if dist < 0 else 1

                # Calculate the distance traveled while accelerating to max_speed
                time_to_max_speed = (self.max_speed - abs(current_speed)) / self.max_acceleration
                distance_accelerating = (
                    time_to_max_speed * current_speed
                    + 0.5 * sgn_factor * self.max_acceleration * time_to_max_speed**2
                )

                # Calculate the distance traveled while deaccelerating to 0
                time_to_slow_down = self.max_speed / self.max_acceleration
                distance_breaking = (
                    sgn_factor * self.max_speed * time_to_slow_down
                    - 0.5 * sgn_factor * self.max_acceleration * time_to_slow_down**2
                )

                # check if elevator has enough distance to reach max_speed
                acceleration_distance = distance_accelerating + distance_breaking
                if abs(acceleration_distance) <= abs(dist):
                    # if yes how long with max speed?
                    distance_with_max_speed = abs(dist) - abs(acceleration_distance)
                    time_with_max_speed = distance_with_max_speed / self.max_speed

                    trajetory_step1 = self.Trajectory(
                        current_pos + distance_accelerating,
                        sgn_factor * self.max_speed,
                        time_to_max_speed,
                    )
                    trajetory_step2 = self.Trajectory(
                        current_pos + distance_accelerating + sgn_factor * distance_with_max_speed,
                        sgn_factor * self.max_speed,
                        time_with_max_speed,
                    )
                    trajetory_step3 = self.Trajectory(
                        current_pos + acceleration_distance + sgn_factor * distance_with_max_speed,
                        0,
                        time_to_slow_down,
                    )

                    self.trajectory_list.extend([trajetory_step1, trajetory_step2, trajetory_step3])
                else:
                    # check if going to overshoot the target?
                    time_to_slow_down = abs(current_speed / self.max_acceleration)
                    distance_slow_down = (
                        current_speed * time_to_slow_down
                        - 0.5 * sgn_factor * self.max_acceleration * time_to_slow_down**2
                    )

                    if abs(distance_slow_down) > abs(dist):
                        # is going to overshoot, stop complety after the target and turn around (next iteration of while)
                        self.trajectory_list.append(
                            self.Trajectory(
                                current_pos + distance_slow_down,
                                0,
                                time_to_slow_down,
                            )
                        )

                    else:
                        # only other option: not enough time to speed up to max_speed
                        # annoying math and quite error prone (yikes)
                        # then calculate best trajectory

                        rem_distance = dist - distance_slow_down
                        # PQ formel to solve equation
                        p = 4 * abs(current_speed) / self.max_acceleration
                        q = 4 * -abs(rem_distance) / self.max_acceleration

                        time_to_speed_up = (-(p / 2) + sqrt((p / 2) ** 2 - q)) / 2

                        final_speed = current_speed + time_to_speed_up * self.max_acceleration * sgn_factor
                        self.trajectory_list.append(
                            self.Trajectory(
                                current_pos + rem_distance / 2,
                                final_speed,
                                time_to_speed_up,
                            )
                        )

                        time_to_slow_down += time_to_speed_up

                        self.trajectory_list.append(
                            self.Trajectory(
                                current_pos + rem_distance + distance_slow_down,
                                final_speed - sgn_factor * time_to_slow_down * self.max_acceleration,
                                time_to_slow_down,
                            )
                        )
        # target position was reached!
        if (next_movement != 0) or self.buttons[int(new_target_position)] == 1:
            self.trajectory_list.append(
                self.trajectory_list[-1]
                .copy()
                .set_open(DOORS_OPEN)
                .set_doors_opening_direction(1)
                .set_time(DOOR_OPENING_TIME)
                .set_next_movement(0)
            )
            self.trajectory_list.append(
                self.trajectory_list[-1]
                .copy()
                .set_open(DOORS_OPEN)
                .set_time(DOOR_STAYING_OPEN_TIME / 2)
                .set_doors_opening_direction(0)
                .set_next_movement(next_movement)
            )
        self.target_position = new_target_position

    def get_time_to_target(self) -> float:
        """Outputs the time this elevator needs until it arrives at the required target plus the time for opening the door

        Returns:
            float: time in seconds

        """
        # if at target position: time to target is infty. Makes sense as we only care about
        # elevators that are not yet at their target
        if len(self.trajectory_list) == 1:
            return INFTY
        # if the elevator is becoming idle in a position, it does not need a decision
        if (
            self.trajectory_list[-1].doors_open == 0
            and self.trajectory_list[-1].doors_open_direction == -1
            and self.trajectory_list[0].next_movement == 0
            and self.target_position == self.trajectory_list[-1].position
        ):
            return INFTY
        for i, t in enumerate(self.trajectory_list):
            if t.next_movement != 0 and t.time > 0:
                return sum([trajectory_step.time for trajectory_step in self.trajectory_list[: i + 1]])
        return sum([trajectory_step.time for trajectory_step in self.trajectory_list])

    def advance_simulation(self, time_step: float) -> bool:
        """Moves the elevator alongs its calculated trajectory until time step is up or the trajectory is completed.

        Args:
            time_step (float): the time to advance the simulation in seconds.

        Returns:
            (bool): that tells whether an elevator has arrived at a floor

        """
        # if only one element in trajectory list => elevator at target position. Do not move
        assert time_step >= 0
        if time_step == 0:
            return False
        if len(self.trajectory_list) == 1:
            return False

        # find first step of trajectory with cummulated simulation time more than time_step
        i = 0
        while i + 1 < len(self.trajectory_list) and time_step - self.trajectory_list[i + 1].time >= 0:
            time_step -= self.trajectory_list[i + 1].time
            i += 1

        if i + 1 == len(self.trajectory_list):
            self.trajectory_list = [self.trajectory_list[-1].copy().set_time(0)]
            return True

        # did the door close before starting to move?

        self.trajectory_list = self.trajectory_list[i:]

        # modify partial executed step along trajectory
        last_pos, last_speed, last_time, last_doors = self.trajectory_list[0].get_values()
        next_pos, next_speed, next_time, next_doors = self.trajectory_list[1].get_values()

        percentage_excecuted = time_step / next_time

        new_doors = last_doors + (next_doors - last_doors) * percentage_excecuted
        new_speed = last_speed + (next_speed - last_speed) * percentage_excecuted
        new_pos = last_pos + last_speed * time_step + (new_speed - last_speed) / 2 * time_step
        new_time = next_time - time_step
        new_doors_direction = sign(next_doors - last_doors)
        new_next_movement = self.trajectory_list[1].next_movement

        self.trajectory_list[0] = self.Trajectory(
            new_pos,
            new_speed,
            0,
            doors_open=new_doors,
            doors_open_direction=new_doors_direction,
        ).set_next_movement(new_next_movement)
        self.trajectory_list[1] = (
            self.trajectory_list[1].copy().set_time(new_time).set_doors_opening_direction(new_doors_direction)
        )

        # elevator has not arrived
        return False

    def get_position(self) -> float:
        """Returns the current position of the elevator

        Returns:
            float: the position in floors
        """
        return self.trajectory_list[0].position

    def get_speed(self) -> float:
        """Returns the current speed of the elevator

        Returns:
            float: the speed (positive going up / negative going down) in floors per second
        """
        return self.trajectory_list[0].speed

    def get_doors_open(self) -> float:
        """Returns the current "openness" value of the doors.

        Returns:
            float: the percentage of how much the doors are open (1 = open,  0 = closed)
        """
        return self.trajectory_list[0].doors_open

    def are_doors_opening(self) -> bool:
        return self.trajectory_list[0].are_doors_opening()

    def set_doors_open(self, new_percentage_open: float):
        """Set the doors_open value for the elevator.

        Args:
            new_percentage_open (float): the percentage of how much the doors are open (1 = open,  0 = closed)

        Raises:
            Exception: If tried to set while on a trajectory.
        """
        if len(self.trajectory_list) > 1:
            # do not open doors midpath
            raise Exception("Cannot set doors_open value while on a trajectory. Must have arrived at target.")
        self.trajectory_list[0].set_open(new_percentage_open)

    def get_target_position(self) -> int:
        """Returns the target floor for this elevator.

        Returns:
            int: the value of the target floor (0 = ground floor, num_floors - 1 = highest floor)
        """
        return self.target_position

    def is_waiting_for_people(self) -> bool:
        """Test if elevator is currently waiting for someone. I.e. stands still doesnt move and doors are open, but is not idle."""
        # if len(self.trajectory_list) < 2:
        #     return False
        # return (
        #     self.trajectory_list[0].position == self.trajectory_list[1].position
        #     and self.trajectory_list[0].speed == self.trajectory_list[1].speed
        #     and self.trajectory_list[0].doors_open == self.trajectory_list[1].doors_open
        #     and self.trajectory_list[1].doors_open_direction == 0
        #     and self.trajectory_list[1].time > 0
        # )
        return self.trajectory_list[0].next_movement != 0
