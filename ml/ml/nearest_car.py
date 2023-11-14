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

    def decide(self, observations, error):
        return self.scheduler_nearest_car(elev_positions=observations["position"], buttons_out=observations["floors"],
                                          buttons_in=observations["buttons"], elev_speed=observations["speed"], max_acceleration=self.max_acceleration)

    def scheduler_nearest_car(self, elev_positions, buttons_out, buttons_in, elev_speed, max_acceleration):

        res = self.scheduler_nearest_car_helper(elev_positions, buttons_out, buttons_in, elev_speed, max_acceleration)

        # depress buttons out
        new_buttons_out = np.zeros_like(buttons_out)
        # modify buttons insde

        # set has seve array inside the button array
        new_buttons_in = buttons_in.copy()
        for t_idx, ts in enumerate(res["to_serve"]):
            for m in ts:
                new_buttons_in[t_idx][m] = True

        for elev_idx, bi in enumerate(buttons_in):
            # turn of the current floor
            # TODO FIX, added int() around "res["target"][elev_idx]" to make it run
            elev_current_floor = int(res["target"][elev_idx])
            new_buttons_in[elev_idx][elev_current_floor] = False

        new_speed = [0] * len(elev_positions)

        next = self.scheduler_nearest_car_helper(
            res["target"], new_buttons_out, new_buttons_in, new_speed, max_acceleration)

        directions = np.array([-1] * len(next["target"]))
        for i in range(0, len(next["target"])):
            d = next["target"][i] - res["target"][i]

            # normalize direction
            if d != 0:
                d = d / abs(d)  # -2 -> -1
            directions[i] = d

        return {"target": res["target"], "to_serve": directions}

    def scheduler_nearest_car_helper(self, elev_positions, buttons_out, buttons_in, elev_speed, max_acceleration):
        calls = []

        # create an algorithm specific format
        for floor_it, b in enumerate(buttons_out):
            if b[0] == True:
                calls.append({"type": "up", "floor": floor_it, "elevator": None})

            if b[1] == True:
                calls.append({"type": "down", "floor": floor_it, "elevator": None})

        for call in calls:
            fs_score = []
            for elev_it, elev_pos in enumerate(elev_positions):

                d = abs(elev_pos - call["floor"])
                elev_dir = self.calc_elev_dir(elev_speed[elev_it])

                fs = -1

                # check if elev can serve the call
                if (not self.can_serve(call["floor"], elev_pos, elev_speed[elev_it])):
                    fs = -2

                # check if elevator is moving away from call
                elif ((elev_pos - call["floor"]) * elev_dir > 1):
                    fs = 1

                # at this point elevator is moving towards us. Check if same direction as call
                elif (elev_dir == 1 and call["type"] == "up") or (elev_dir == -1 and call["type"] == "down"):
                    fs = (self.num_floors + 2) - d

                # check if direction is different from call
                elif (elev_dir == 1 and call["type"] == "down") or (elev_dir == -1 and call["type"] == "up"):
                    fs = (self.num_floors + 1) - d

                # check if elevator is idle
                elif (elev_dir == 0):
                    fs = (self.num_floors + 1) - d
                else:
                    raise Exception("Should not happen")

                fs_score.append(fs)
                # print("Call at Floor:", call["floor"], " Elevator", elev_it, "\tFS:", fs,"\tN", N,"\td", d)

            # Assign to the call the elevator with the max fs_score
            max_fs = -np.inf
            max_elevator = None
            for fs_it, f in enumerate(fs_score):

                if f > max_fs:
                    max_elevator = fs_it
                    max_fs = f
                # break tie
                elif f == max_fs:
                    if abs(elev_positions[fs_it] - call["floor"]) < abs(elev_positions[max_elevator] - call["floor"]):
                        max_elevator = fs_it
                        max_fs = f

            # print("Call gets served by:", max_elevator)

            call["elevator"] = max_elevator

        target_result = []
        to_serve_result = []

        # calculate the target floor for each elevator
        for elev_it, elev_pos in enumerate(elev_positions):
            to_serve = set()

            # elevator need to serve calls outside
            for call in calls:
                if call["elevator"] == elev_it:
                    to_serve.add(call["floor"])

            for inside_call_it, inside_call in enumerate(buttons_in[elev_it]):
                if inside_call == True:
                    to_serve.add(inside_call_it)

            # print("Elevator ", elev_it, "serves: ", to_serve)

            # if elevator has no call, then stopp at nearest one you can stopp at

            to_serve_result.append(to_serve)
            target = self.get_nearest_floor_to_serve(to_serve, elev_pos, elev_it, elev_speed)

            if (target == -1):

                elev_dir = self.calc_elev_dir(elev_speed[elev_it])

                # the elevator has no call. Search nearest waiting floor in movement direction. If not moving then wait on current
                if elev_dir == 0:
                    target = int(elev_pos)
                elif elev_dir == 1:
                    stopps_in_dir = range(int(np.ceil(elev_pos)), (self.num_floors - 1))
                    target = self.get_nearest_floor_to_serve(
                        stopps_in_dir, elev_pos, elev_it, elev_speed)
                elif elev_dir == -1:
                    stopps_in_dir = range(0, int(np.floor(elev_pos)))
                    target = self.get_nearest_floor_to_serve(
                        stopps_in_dir, elev_pos, elev_it, elev_speed)

                # still no target found, then search on all floors
                if target == -1:
                    stopps_in_dir = range(0, (self.num_floors - 1))
                    target = self.get_nearest_floor_to_serve(
                        stopps_in_dir, elev_pos, elev_it, elev_speed)

            target_result.append(target)

        return {"target": np.array(target_result), "to_serve": np.array(to_serve_result)}

    def get_nearest_floor_to_serve(self, calling_floors, elev_pos, elev_it, elev_speed):
        # check which floor the elevator has to serve is the nearet one and can be served
        nearest_floor = -1
        nearest_floor_distance = -1

        for fts in calling_floors:
            if not self.can_serve(fts, elev_pos, elev_speed[elev_it]):
                continue

            d = abs(elev_pos - fts)
            # no initial set
            if nearest_floor_distance == -1:
                nearest_floor_distance = d
                nearest_floor = fts
                continue

            # new best found
            if nearest_floor_distance > d:
                nearest_floor_distance = d
                nearest_floor = fts

        return nearest_floor

    def calc_elev_dir(self, speed):
        return np.sign(speed)

    def can_serve(self, call_floor, elev_pos, elev_speed):
        distance = abs(call_floor - elev_pos)
        min_distance_needed = (elev_speed ** 2) / (2 * self.max_acceleration)
        return min_distance_needed <= distance
