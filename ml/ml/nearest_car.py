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

    def decide(self, observations, error, info):
        elev_positions = observations["elevator"]["position"]
        buttons_out = observations["floors"]
        buttons_in = observations["elevator"]["buttons"]
        elev_speed = observations["elevator"]["speed"]
        result = scheduler_nearest_car(elev_positions=elev_positions, buttons_out=buttons_out,
                              buttons_in=buttons_in, elev_speed=elev_speed, max_acceleration=self.max_acceleration)
        return {"target": result, "next_movement": np.zeros(self.num_elevators)}


def scheduler_nearest_car(elev_positions, buttons_out, buttons_in, elev_speed, max_acceleration):
    calls = []
    N = len(buttons_out)

    # create an algorithm specific format
    for floor_it, b in enumerate(buttons_out):
        if b["up"] == True:
            calls.append({"type": "up", "floor": floor_it, "elevator": None})

        if b["down"] == True:
            calls.append({"type": "down", "floor": floor_it, "elevator": None})

    for call in calls:
        fs_score = []
        for elev_it, elev_pos in enumerate(elev_positions):

            d = abs(elev_pos - call["floor"])
            elev_dir = calc_elev_dir(elev_speed[elev_it])

            fs = -1
            # check if elev can serve the call
            if (not can_serve(call["floor"], elev_pos, elev_speed[elev_it], max_acceleration)):
                fs = -2

            # check if elevator is moving away from call
            elif (elev_pos > call["floor"] and elev_dir == 1) or (elev_pos < call["floor"] and elev_dir == -1):
                fs = 1

            # at this point elevator is moving towards us. Check if same direction as call
            elif (elev_dir == 1 and call["type"] == "up") or (elev_dir == -1 and call["type"] == "down"):
                fs = (N + 2) - d

            # check if direction is different from call
            elif (elev_dir == 1 and call["type"] == "down") or (elev_dir == -1 and call["type"] == "up"):
                fs = (N + 1) - d

            fs_score.append(fs)
            # print("Call at Floor:", call["floor"], " Elevator", elev_it, "\tFS:", fs,"\tN", N,"\td", d)

        # Assign to the call the elevator with the max fs_score
        max_fs = -1
        max_elevator = -1
        for fs_it, f in enumerate(fs_score):

            if f > max_fs:
                max_elevator = fs_it
                max_fs = f
            if f == max_fs:

                if abs(elev_positions[fs_it] - call["floor"]) < abs(elev_positions[max_elevator] - call["floor"]):
                    max_elevator = fs_it
                    max_fs = f

        # print("Call gets served by:", max_elevator)

        call["elevator"] = max_elevator

    result = []

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

        target = get_nearest_floor_to_serve(to_serve, elev_pos, elev_it, elev_speed, max_acceleration)

        if (target == -1):

            elev_dir = calc_elev_dir(elev_speed[elev_it])

            # the elevator has no call. Search nearest waiting floor in movement direction. If not moving then wait on current
            if elev_dir == 0:
                target = elev_pos
            elif elev_dir == 1:
                stopps_in_dir = range(elev_pos, N)
                target = get_nearest_floor_to_serve(stopps_in_dir, elev_pos, elev_it, elev_speed, max_acceleration)
            elif elev_dir == -1:
                stopps_in_dir = range(0, elev_pos)
                target = get_nearest_floor_to_serve(stopps_in_dir, elev_pos, elev_it, elev_speed, max_acceleration)

            # still no target found, then search on all floors
            if target == -1:
                stopps_in_dir = range(0, N)
                target = get_nearest_floor_to_serve(stopps_in_dir, elev_pos, elev_it, elev_speed, max_acceleration)

        result.append(target)

    print(result)

    return result


def get_nearest_floor_to_serve(calling_floors, elev_pos, elev_it, elev_speed, max_acceleration):
    # check which floor the elevator has to serve is the nearet one and can be served
    nearest_floor = -1
    nearest_floor_distance = -1

    for fts in calling_floors:
        if not can_serve(fts, elev_pos, elev_speed[elev_it], max_acceleration):
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


def calc_elev_dir(speed):
    elev_dir = 0
    if speed != 0:
        elev_dir = speed / abs(speed)  # 1, 0, -1
    assert elev_dir in [-1, 0, 1]
    return elev_dir


def can_serve(call_floor, elev_pos, elev_speed, max_acceleration):
    distance = abs(call_floor - elev_pos)

    if max_acceleration == 0:
        return False

    min_distance_needed = (elev_speed * elev_speed) / (2 * max_acceleration)

    if min_distance_needed > distance:
        return False
    return True


# TESTS
current_tests = []

if 1 in current_tests:
    test1 = {
        "elev_pos": [1, 3],
        "elev_speed": [4.6, -3.2],
        "buttons_out": [{"up": True, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": True, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "True": "Time", "down": False, "down_time": "Time"},
                        ],
        "buttons_in": [[False, False, False, True], [False, True, False, False]]
    }

    elev_vis.print_elevator(test1["elev_pos"], test1["buttons_out"], test1["buttons_in"], test1["elev_speed"])
    t = scheduler_nearest_car(test1["elev_pos"], test1["buttons_out"], test1["buttons_in"], test1["elev_speed"], 1000)
    assert t == [3, 1]
    print("--------------------------------")

if 2 in current_tests:
    test2 = {
        "elev_pos": [2],
        "elev_speed": [-1],
        "buttons_out": [{"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "True": "Time", "down": False, "down_time": "Time"},
                        ],
        "buttons_in": [[False, False, False, False, False]]
    }

    elev_vis.print_elevator(test2["elev_pos"], test2["buttons_out"], test2["buttons_in"], test2["elev_speed"])
    t = scheduler_nearest_car(test2["elev_pos"], test2["buttons_out"], test2["buttons_in"], test2["elev_speed"], 1000)
    print("Test 2", t)
    assert t == [1]
    print("--------------------------------")

if 3 in current_tests:
    test3 = {
        "elev_pos": [2, 0, 3, 1],
        "elev_speed": [2, 2, 0, 3],
        "buttons_out": [{"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": True, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "up_time": "Time", "down": False, "down_time": "Time"},
                        {"up": False, "True": "Time", "down": False, "down_time": "Time"},
                        ],
        "buttons_in": [[False, False, False, False, False], [False, False, True, False, False],
                       [False, False, False, False, True], [False, False, True, True, False]]
    }

    elev_vis.print_elevator(test3["elev_pos"], test3["buttons_out"], test3["buttons_in"], test3["elev_speed"])
    scheduler_nearest_car(test3["elev_pos"], test3["buttons_out"], test3["buttons_in"], test3["elev_speed"], 1000)
