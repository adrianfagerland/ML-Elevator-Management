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


class Elevator:
    def __init__(
        self,
        elevator_number,
        current_postion=0,
        target=None,
        next_move=0,
        buttons_inside=[],
        speed=0,
        door=[],
        max_acceleration=1,
    ) -> None:
        self.speed = speed
        self.number = elevator_number
        self.direction = self.update_direction()
        self.buttons_inside = buttons_inside
        self.floors_next_move = self.buttons_inside
        self.position: float = current_postion
        self.max_acceleration = max_acceleration
        self.door: float = door
        self.next_move = next_move
        self.door_moving_direction = 0
        if target is None:
            self.target = current_postion

    def can_serve(self, position):
        distance = abs(position - self.position)
        min_distance_needed = (self.speed**2) / (2 * self.max_acceleration)
        return min_distance_needed - 0.000001 <= distance

    def update_direction(self):
        return 0 if self.speed == 0 else self.speed / abs(self.speed)

    def call_is_on_route(self, call):
        return (
            sum(self.buttons_inside) == 0
            or (self.position <= call["floor"] <= self.target and call["direction"] == 1)
            or (self.position >= call["floor"] >= self.target and call["direction"] == -1)
        )


class NearestCar(Scheduler):
    def __init__(self, num_elevators, num_floors, max_speed, max_acceleration) -> None:
        super().__init__(num_elevators, num_floors, max_speed, max_acceleration)
        self.elevators = [
            Elevator(
                elevator_number=i,
                current_postion=0,
                max_acceleration=max_acceleration,
                buttons_inside=[0] * num_floors,
                door=0,
            )
            for i in range(num_elevators)
        ]
        self.N = num_floors

    def decide(self, observations, error):
        self.update_elevators(observations)
        calls = self.floors_to_calls(observations["floors"])
        return self.scheduler_nearest_car(calls)

    def scheduler_nearest_car(self, calls):
        for elevator in self.elevators:
            potential_target_floors = [i for i, x in enumerate(elevator.buttons_inside) if x == 1]
            if len(potential_target_floors) > 0:
                closest_floor = min(potential_target_floors, key=lambda x: abs(x - elevator.position))
                elevator.target = closest_floor

        # calculate for every call which elevator will serve it
        self.evaluate_calls(calls)

        return {
            "target": [e.target for e in self.elevators],
            "next_move": [e.next_move for e in self.elevators],
        }

    def evaluate_calls(self, calls):
        busy_elevators = {}
        call_stack = calls
        while len(call_stack) > 0:
            call = call_stack.pop()
            fs_values = [self.calc_fs(call["direction"], call["floor"], elev) for elev in self.elevators]
            # cycle through self.elevators using the decreasing fs value
            for elev in sorted(self.elevators, key=lambda x: fs_values[x.number], reverse=True):
                if elev.call_is_on_route(call):
                    if elev in busy_elevators and busy_elevators[elev][1] < fs_values[elev.number]:
                        call_stack.append(busy_elevators[elev][0])
                        busy_elevators[elev] = (call, fs_values[elev.number])
                        break
                    elif elev not in busy_elevators:
                        busy_elevators[elev] = (call, fs_values[elev.number])
                        break
        for elev in self.elevators:
            if elev in busy_elevators:
                call = busy_elevators[elev][0]
                elev.target = call["floor"]
                elev.next_move = call["direction"]
            else:
                elev.next_move = 0

    def calc_fs(self, call_direction, call_floor, elevator: Elevator):
        distance = abs(elevator.position - call_floor)

        # check if elevator can not serve floor RET0
        if not elevator.can_serve(call_floor):
            fs = 0

        # elevator is moving away from call RER1
        elif (elevator.direction == 1 and elevator.position > call_floor) or (
            elevator.direction == -1 and elevator.position < call_floor
        ):
            fs = 1

        # # elevator is not moving RET 1.1 TODO decide on this value
        elif elevator.direction == 0:
            fs = self.N + 1.5 - distance - 0.1 * elevator.door

        # vvv at this point elevator is moving towards call vvv
        # elif direction of elevator the same as call direction RET (N+2)-d
        elif elevator.direction == call_direction:
            fs = self.N + 2 - distance

        # elif direction of elevator the oposit as call direction RET (N+1)-d
        elif elevator.direction == call_direction * -1:
            fs = self.N + 1 - distance
        else:
            raise Exception("Something went wrong")

        return fs

    def floors_to_calls(self, floors):
        out = []
        for i in range(0, len(floors)):
            for j in range(0, len(floors[0])):
                if floors[i][j] == 1 and j == 0:
                    out.append({"direction": 1, "floor": i})
                if floors[i][j] == 1 and j == 1:
                    out.append({"direction": -1, "floor": i})
        return out

    def update_elevators(self, observations):
        for i, elevator in enumerate(self.elevators):
            elevator.position = observations["elevators"][i]["position"][0]
            elevator.speed = observations["elevators"][i]["speed"][0]
            elevator.update_direction()
            elevator.buttons_inside = observations["elevators"][i]["buttons"]
            elevator.door = observations["elevators"][i]["doors_state"][0]
            elevator.door_moving_direction = observations["elevators"][i]["doors_moving_direction"][0]
            elevator.target = observations["elevators"][i]["target"]
