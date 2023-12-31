import numpy as np
from elsim.elevator_simulator import ElevatorSimulator


def _compare_elevator_state_from_observation(
    observation1,
    observation2,
    floors_should_be_equal=None,
    time_should_be_equal=None,
    num_elevators_should_be_equal=None,
    position_should_be_equal=None,
    speed_should_be_equal=None,
    buttons_should_be_equal=None,
    doors_state_should_be_equal=None,
    doors_moving_direction_should_be_equal=None,
    elevators_to_check=None,
):
    assert observation1[-1]["needs_decision"] == True
    observation1 = observation1[0]
    observation2 = observation2[0]
    num_elevators = observation1["num_elevators"][0]

    if floors_should_be_equal is not None:
        assert floors_should_be_equal == np.array_equal(observation1["floors"], observation2["floors"])
    if time_should_be_equal is not None:
        assert time_should_be_equal == (observation1["time"] == observation2["time"])
    if num_elevators_should_be_equal is not None:
        assert num_elevators_should_be_equal == (observation1["num_elevators"][0] == observation2["num_elevators"][0])
    for i in range(num_elevators):
        if elevators_to_check is not None and elevators_to_check[i] == 0:
            continue
        if position_should_be_equal is not None:
            assert position_should_be_equal == np.array_equal(
                observation1["elevators"][i]["position"], observation2["elevators"][i]["position"]
            )
        if speed_should_be_equal is not None:
            assert speed_should_be_equal == np.array_equal(
                observation1["elevators"][i]["speed"], observation2["elevators"][i]["speed"]
            )
        if buttons_should_be_equal is not None:
            assert buttons_should_be_equal == np.array_equal(
                observation1["elevators"][i]["buttons"], observation2["elevators"][i]["buttons"]
            )
        if doors_state_should_be_equal is not None:
            assert doors_state_should_be_equal == np.array_equal(
                observation1["elevators"][i]["doors_state"], observation2["elevators"][i]["doors_state"]
            )
        if doors_moving_direction_should_be_equal is not None:
            assert doors_moving_direction_should_be_equal == np.array_equal(
                observation1["elevators"][i]["doors_moving_direction"],
                observation2["elevators"][i]["doors_moving_direction"],
            )


def test_get_number_of_people_in_sim_few():
    simulator = ElevatorSimulator(num_floors=4, num_elevators=1, num_arrivals=3)
    simulator.init_simulation(density=1, seed=42)
    simulator.step(None)
    assert simulator.get_number_of_people_in_sim() == 1
    simulator.step(None)
    assert simulator.get_number_of_people_in_sim() == 2
    simulator.step(None)
    assert simulator.get_number_of_people_in_sim() == 3
    simulator.step(None)
    assert simulator.get_number_of_people_in_sim() == 3


def test_get_number_of_people_in_sim_many():
    simulator = ElevatorSimulator(num_floors=100, num_elevators=10, num_arrivals=3)
    simulator.init_simulation(density=1, seed=42)
    # they all arrive at the same time
    simulator.step(None)
    assert simulator.get_number_of_people_in_sim() == 3


def test_step_no_actions():
    simulator = ElevatorSimulator(num_floors=100, num_elevators=10, num_arrivals=3)
    simulator.init_simulation(density=1, seed=42)
    observation1 = simulator.get_observations()
    simulator.step(None)
    observation2 = simulator.get_observations()
    _compare_elevator_state_from_observation(
        observation1,
        observation2,
        floors_should_be_equal=False,
        time_should_be_equal=False,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
    )


def test_step_with_actions():
    simulator = ElevatorSimulator(num_floors=30, num_elevators=2, num_arrivals=3)
    simulator.init_simulation(density=1, seed=42)
    observation1 = simulator.get_observations()
    targets = np.array([e["position"] for e in observation1[0]["elevators"]])
    targets[0] = 1
    next_moves = [0 for _ in observation1[0]["elevators"]]
    actions = [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    observation2 = simulator.step(actions)
    # the next step that happens is not that the elevator arrives at the target, but that someone arrives at a floor
    _compare_elevator_state_from_observation(
        observation1,
        observation2,
        elevators_to_check=[1] + [0 for _ in range(1, observation1[0]["num_elevators"][0])],
        position_should_be_equal=False,
        speed_should_be_equal=False,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
    )
    _compare_elevator_state_from_observation(
        observation1,
        observation2,
        floors_should_be_equal=False,
        time_should_be_equal=False,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
        elevators_to_check=[0] + [1 for _ in range(1, observation1[0]["num_elevators"][0])],
    )


def test_deep_situation_action1():
    simulator = ElevatorSimulator(num_floors=10, num_elevators=3, num_arrivals=3)
    simulator.init_simulation(density=1, seed=2002)
    simulator.generate_arrivals_data(seed=42)
    observation1 = simulator.step(None)
    # assert that the elevator state is at is. quite important for the rest of the test
    correct_observation1 = (
        {
            "floors": np.array(list(zip([0] * 10, [0] * 9 + [1])), dtype=np.int8),
            "num_elevators": np.array([3], dtype=np.uint8),
            "time": {
                "time_seconds": np.array([2.0], dtype=np.float32),
                "time_since_last_seconds": np.array([2.0], dtype=np.float32),
            },
            "elevators": tuple(
                [
                    {
                        "position": np.array([pos], dtype=np.float32),
                        "speed": np.array([0.0], dtype=np.float32),
                        "buttons": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        "doors_state": np.array([0.0], dtype=np.float32),
                        "doors_moving_direction": np.array([0.0], dtype=np.float32),
                    }
                    for pos in [0.0, 8.0, 5.0]
                ]
            ),
        },
    )
    _compare_elevator_state_from_observation(
        observation1,
        correct_observation1,
        floors_should_be_equal=True,
        time_should_be_equal=True,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
    )
    targets = np.array([0, 9, 5])
    next_moves = np.array([0, -1, 0])
    observation2 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    _compare_elevator_state_from_observation(
        observation1,
        observation2,
        elevators_to_check=[0, 1, 0],
        position_should_be_equal=False,
        speed_should_be_equal=False,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
    )
    _compare_elevator_state_from_observation(
        observation1,
        observation2,
        floors_should_be_equal=False,
        time_should_be_equal=False,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
        elevators_to_check=[1, 0, 1],
    )
    targets = np.array([0, 9, 5])
    next_moves = np.array([0, -1, 0])
    observation3 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    _compare_elevator_state_from_observation(
        observation2,
        observation3,
        elevators_to_check=[0, 1, 0],
        position_should_be_equal=False,
        speed_should_be_equal=False,
        buttons_should_be_equal=False,
        doors_state_should_be_equal=False,
        doors_moving_direction_should_be_equal=True,
    )
    assert observation3[0]["elevators"][1]["doors_state"] == 1
    _compare_elevator_state_from_observation(
        observation2,
        observation3,
        floors_should_be_equal=False,
        time_should_be_equal=False,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
        elevators_to_check=[1, 0, 1],
    )
    targets = np.array([0, 0, 5])
    next_moves = np.array([0, 0, 0])
    observation4 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    _compare_elevator_state_from_observation(
        observation3,
        observation4,
        elevators_to_check=[0, 1, 0],
        position_should_be_equal=False,
        speed_should_be_equal=False,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=False,
        doors_moving_direction_should_be_equal=True,
    )
    _compare_elevator_state_from_observation(
        observation3,
        observation4,
        floors_should_be_equal=False,
        time_should_be_equal=False,
        num_elevators_should_be_equal=True,
        position_should_be_equal=True,
        speed_should_be_equal=True,
        buttons_should_be_equal=True,
        doors_state_should_be_equal=True,
        doors_moving_direction_should_be_equal=True,
        elevators_to_check=[1, 0, 1],
    )
    assert observation4[2] == False
    targets = np.array([0, 0, 5])
    next_moves = np.array([0, 0, 0])
    observation5 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    targets = np.array([2, 0, 6])
    next_moves = np.array([-1, 0, -1])
    observation6 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    targets = np.array([2, 0, 0])
    next_moves = np.array([-1, 0, 0])
    observation7 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    targets = np.array([0, 0, 0])
    next_moves = np.array([0, 0, 0])
    observation8 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    observation9 = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    last_observation = simulator.step(
        [{"target": target, "next_move": next_move} for target, next_move in zip(targets, next_moves)]
    )
    assert simulator.get_number_of_people_in_sim() == 0
    assert last_observation[2] == True
