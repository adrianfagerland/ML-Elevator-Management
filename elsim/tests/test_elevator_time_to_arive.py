from elsim.elevator_simulator import ElevatorSimulator
from elsim.elevator import Elevator
from elsim.parameters import DOOR_STAYING_OPEN_TIME, INFTY, DOOR_OPENING_TIME
####################
# TEST ARRIVAL TIME
####################


def test_time_with_partial_door_openings():
    test_elevator = Elevator(0, 11, 2, 1)
    test_elevator.set_target_position(10)
    assert test_elevator.get_time_to_target() == 7 + 1 * DOOR_OPENING_TIME + \
        DOOR_STAYING_OPEN_TIME

    # Test if door needs to be closed
    test_elevator = Elevator(0, 11, 2, 1)
    test_elevator.set_doors_open(1)
    test_elevator.set_target_position(10)
    # Test open door

    assert test_elevator.get_time_to_target() == 7 + 2 * DOOR_OPENING_TIME + \
        DOOR_STAYING_OPEN_TIME


####################
# TEST TRAJECTORYS
####################

def test_normal_arival_with_full_speed():
    test_elevator = Elevator(0, 11, 2, 1)
    test_elevator.set_target_position(10)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(0.0, 0.0, 0.0),
                                             Elevator.Trajectory(
                                                 2.0, 2.0, 2.0),
                                             Elevator.Trajectory(
                                                 8.0, 2.0, 3.0),
                                             Elevator.Trajectory(
        10.0, 0.0, 2.0),
        Elevator.Trajectory(
        10.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        10.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]

    test_elevator = Elevator(10, 11, 2, 1)
    test_elevator.set_target_position(0)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(10.0, 0.0, 0.0),
                                             Elevator.Trajectory(
                                                 8.0, -2.0, 2.0),
                                             Elevator.Trajectory(
                                                 2.0, -2.0, 3.0),
                                             Elevator.Trajectory(
        0.0, 0.0, 2.0),
        Elevator.Trajectory(
        0.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        0.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]


def test_reverse_speed():
    test_elevator = Elevator(3, 11, 2, 1, current_speed=-1)
    test_elevator.set_target_position(10)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(3, -1, 0),
                                             Elevator.Trajectory(2.5, 0, 1.0),
                                             Elevator.Trajectory(4.5, 2, 2.0),
                                             Elevator.Trajectory(8.0, 2, 1.75),
                                             Elevator.Trajectory(
        10.0, 0, 2.0),
        Elevator.Trajectory(
        10.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        10.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]

    test_elevator = Elevator(7, 11, 2, 1, current_speed=1)
    test_elevator.set_target_position(0)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(7, 1, 0),
                                             Elevator.Trajectory(7.5, 0, 1.0),
                                             Elevator.Trajectory(5.5, -2, 2.0),
                                             Elevator.Trajectory(
                                                 2.0, -2, 1.75),
                                             Elevator.Trajectory(0.0, 0, 2.0),
                                             Elevator.Trajectory(
        0.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        0.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]
    pass


def test_not_full_speed():
    test_elevator = Elevator(3 - 1/16, 10, 3, 1, current_speed=1)
    test_elevator.set_target_position(4)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(3 - 1/16, 1, 0),
                                             Elevator.Trajectory(
                                                 3.25 - 1/32, 1.25, 0.25),
                                             Elevator.Trajectory(
        4.0, 0.0, 1.25),
        Elevator.Trajectory(
        4.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        4.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]

    test_elevator = Elevator(1, 10, 6, 2, current_speed=3)
    test_elevator.set_target_position(6)

    assert test_elevator.trajectory_list == [Elevator.Trajectory(1, 3, 0),
                                             Elevator.Trajectory(
                                                 2.375, 3.8078865529319543, 0.40394327646597716),
                                             Elevator.Trajectory(
        6.0, 0.0, 1.9039432764659772),
        Elevator.Trajectory(
        6.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        6.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]

    test_elevator = Elevator(7 + 1/16, 10, 3, 1, current_speed=-1)
    test_elevator.set_target_position(6)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(7 + 1/16, -1, 0),
                                             Elevator.Trajectory(
                                                 6.75 + 1/32, -1.25, 0.25),
                                             Elevator.Trajectory(
        6.0, 0.0, 1.25),
        Elevator.Trajectory(
        6.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        6.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]


def test_overshot_speed():
    test_elevator = Elevator(3.5, 10, 2, 1, current_speed=2)
    test_elevator.set_target_position(4)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(3.5, 2, 0),
                                             Elevator.Trajectory(
                                                 5.5, 0.0, 2.0),
                                             Elevator.Trajectory(
                                                 4.75, -1.224744871391589, 1.224744871391589),
                                             Elevator.Trajectory(
        4.0, 0.0, 1.224744871391589),
        Elevator.Trajectory(
                                                 4.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        4.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]

    test_elevator = Elevator(4.5, 10, 2, 1, current_speed=-2)
    test_elevator.set_target_position(4)
    assert test_elevator.trajectory_list == [Elevator.Trajectory(4.5, -2, 0),
                                             Elevator.Trajectory(
                                                 2.5, 0.0, 2.0),
                                             Elevator.Trajectory(
                                                 3.25, 1.224744871391589, 1.224744871391589),
                                             Elevator.Trajectory(
        4.0, 0.0, 1.224744871391589),
        Elevator.Trajectory(
                                                 4.0, 0.0, DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
        Elevator.Trajectory(
        4.0, 0.0, DOOR_STAYING_OPEN_TIME, doors_open=1),
    ]
    pass


def test_arrived_elevator():
    test_elevator = Elevator(4, 10, 2, 1)
    assert test_elevator.get_time_to_target() == INFTY
    test_elevator.set_target_position(9)
    t = test_elevator.get_time_to_target()
    test_elevator.advance_simulation(t)
    assert test_elevator.get_time_to_target() == INFTY
