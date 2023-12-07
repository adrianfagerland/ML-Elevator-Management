import numpy as np

from ml.scheduler import Scheduler

# Nearest Car: Compute for every call which elevator should serve it.
# The nearest elevator (depending on the direction) is assigned to call. Then assign to every elevator the nearest floor it
# has to serve
# INPUT
# Position of elevator (Array: [5,2,5,6,0])
# Pressed buttons outside (Array: [{up: False, up_time: Time, down: True, down_time: Time}, ...]
# Pressed buttons inside (Array of Arrays: [[True, False, False], ...]) fist elevator wants to go to floor 0
# Elevator Speed (Array: [100, -40, 10, 0, 100]
# OUTPUT
# Position where to go next (Array: [6,2,5,6,0])


class NearestCar(Scheduler):
    def __init__(self, num_elevators, num_floors, max_speed, max_acceleration) -> None:
        super().__init__(num_elevators, num_floors, max_speed, max_acceleration)
        self.elevators = [Elevator(i, 0, max_acceleration=max_acceleration) for i in range(num_elevators)]

    def decide(self, observations, error):
        self.update_elevators(observations)
        calls = self.floors_to_calls(observations["floors"])
        N = len(observations["floors"])
        return self.scheduler_nearest_car(N, calls)

    def scheduler_nearest_car(self, N, calls):
        target = [elevator.position if abs(elevator.speed) < 0.000001 else [0] for elevator in self.elevators]
        next_move = np.array([0] * 4)

        for elevator in self.elevators:
            potential_target_floors = [i for i, x in enumerate(elevator.buttons_inside) if x == 1]
            elevator.buttons_inside = [0 if elevator.position ==
                                       floor else pressed for floor, pressed in enumerate(elevator.buttons_inside)]
            if len(potential_target_floors) > 0:
                closest_floor = min(potential_target_floors, key=lambda x: abs(x - elevator.position))
                target[elevator.number] = closest_floor

        # calculate for every call which elevator will serve it
        for call in calls:

            best_elevator = self.elevators[0]  # default is 0
            best_fs_value = -999

            for elev in self.elevators:
                fs = self.calc_fs(N, call["direction"], call["floor"], elev)
                if fs > best_fs_value and (sum(elev.buttons_inside) == 0 or (elev.position <= call["floor"] <= target[elev.number]) or (elev.position >= call["floor"] >= target[elev.number])):
                    best_fs_value = fs
                    best_elevator = elev

                # if equally good then take the nearest
                elif fs == best_fs_value and (abs(elev.position - call["floor"]) < abs(best_elevator.position - call["floor"])):
                    best_fs_value = fs
                    best_elevator = elev

            best_elevator.add_call(call)
            target[best_elevator.number] = call["floor"]
            next_move[best_elevator.number] = call["direction"]

            # print("Call", call["direction"], "at floor",call["floor"]," will be served by elevator", best_elevator.number)

        return {"target": target, "next_move": next_move}

    def calc_fs(self, N, call_direction, call_floor, elevator):
        distance = abs(elevator.position - call_floor)

        # check if elevator can not serve floor RET0
        if not elevator.can_serve(call_floor):
            return 0

        # elevator is moving away from call RER1
        if (elevator.direction == 1 and elevator.position > call_floor) or (elevator.direction == -1 and elevator.position < call_floor):
            return 1

        # # elevator is not moving RET 1.1 TODO decide on this value
        if (elevator.direction == 0):
            return N+1.5-distance

        # vvv at this point elevator is moving towards call vvv
        # if direction of elevator the same as call direction RET (N+2)-d
        if (elevator.direction == call_direction):
            return N+2-distance

        # if direction of elevator the oposit as call direction RET (N+1)-d
        if (elevator.direction == call_direction * -1):
            return N+1-distance

        # this should not happen
        raise Exception(
            "[Nearest Car] should not happen")

    def floors_to_calls(self, floors):

        out = []

        for i in range(0, len(floors)):
            for j in range(0, len(floors[0])):
                if floors[i][j] == 1 and j == 0:
                    out.append({"direction": 1, "floor": i})
                    continue

                if floors[i][j] == 1 and j == 1:
                    out.append({"direction": -1, "floor": i})
                    continue

        return out

    def update_elevators(self, observations):
        for i, elevator in enumerate(self.elevators):
            elevator.position = observations["position"][i]
            elevator.speed = observations["speed"][i]
            elevator.update_direction()
            elevator.buttons_inside = observations["buttons"][i]


class Elevator:
    def __init__(self, elevator_number, current_postion=0, buttons_inside=[], speed=0, max_acceleration=9999) -> None:
        self.speed = speed
        self.number = elevator_number
        self.direction = self.update_direction()
        self.buttons_inside = buttons_inside
        self.floors_to_serve = self.buttons_inside
        self.position: float = current_postion
        self.max_acceleration = max_acceleration
        self.calls = []

    def can_serve(self,  position):
        distance = abs(position - self.position)
        min_distance_needed = (self.speed ** 2) / (2 * self.max_acceleration)
        return (min_distance_needed - 0.000001 <= distance)

    def update_direction(self):
        return 0 if self.speed == 0 else self.speed / abs(self.speed)

    def add_call(self, call):
        self.calls.append(call)
