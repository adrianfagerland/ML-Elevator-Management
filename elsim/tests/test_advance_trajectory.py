from elsim.elevator import Elevator
from elsim.parameters import DOOR_STAYING_OPEN_TIME, INFTY, DOOR_OPENING_TIME
import pytest


####################
## TEST ADVANCE
####################
def test_time_simple():
    compare_list = {
        0:[Elevator.Trajectory(0,0,0),
           Elevator.Trajectory(2,2,2),
           Elevator.Trajectory(8,2,3),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        1:[Elevator.Trajectory(0.5,1,0),
           Elevator.Trajectory(2,2,1),
           Elevator.Trajectory(8,2,3),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        2:[Elevator.Trajectory(2,2,0),
           Elevator.Trajectory(8,2,3),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        3:[Elevator.Trajectory(4,2,0),
           Elevator.Trajectory(8,2,2),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        4:[Elevator.Trajectory(6,2,0),
           Elevator.Trajectory(8,2,1),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        5:[Elevator.Trajectory(8,2,0),
           Elevator.Trajectory(10,0,2),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        6:[Elevator.Trajectory(9.5,1,0),
           Elevator.Trajectory(10,0,1),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],
        7:[Elevator.Trajectory(10,0,0),
           Elevator.Trajectory(10,0,DOOR_OPENING_TIME, doors_open=1, doors_open_direction=1),
           Elevator.Trajectory(10,0,DOOR_STAYING_OPEN_TIME, doors_open=1),
           ],

    }
    for i in range(0,8):
        test_elevator =  Elevator(0,11,2,1)
        # set goal and baseline
        test_elevator.set_target_position(10)
        total_time = test_elevator.get_time_to_target()

        # start tests
        test_elevator.advance_simulation(i)
        assert test_elevator.trajectory_list == compare_list[i]
        assert test_elevator.get_time_to_target() == total_time - i

def test_with_door_openings():

   test_elevator = Elevator(4,11,2,1)
   test_elevator.set_doors_open(1)
   test_elevator.set_target_position(10)
   test_elevator.advance_simulation(DOOR_OPENING_TIME / 2)
   test_elevator.set_target_position(0)
   test_elevator.advance_simulation(2)
   t1 = test_elevator.get_time_to_target()

   test_elevator = Elevator(4,11,2,1)
   test_elevator.set_doors_open(1)
   test_elevator.set_target_position(0)
   test_elevator.advance_simulation(2)
   assert test_elevator.get_time_to_target() == t1 + DOOR_OPENING_TIME / 2

def test_empty_trajectory():
   test_elevator = Elevator(4,11,2,1)
   test_elevator.advance_simulation(1000)
   assert test_elevator.get_position() == Elevator(4,11,2,1).get_position()
   assert test_elevator.get_speed() == Elevator(4,11,2,1).get_speed()
   assert test_elevator.get_doors_open() == Elevator(4,11,2,1).get_doors_open()
# TODO write more tests

def test_doors_simple():

   test_elevator = Elevator(4,11,2,1)

   test_elevator.set_doors_open(1)

   test_elevator.set_target_position(8)
   total_time = test_elevator.get_time_to_target()
   test_elevator.advance_simulation(0.1 * DOOR_OPENING_TIME)
   assert pytest.approx(total_time - 0.1*DOOR_OPENING_TIME) == test_elevator.get_time_to_target()
   assert pytest.approx(test_elevator.get_doors_open()) == 0.9

   test_elevator.advance_simulation(0.5 * DOOR_OPENING_TIME)
   assert pytest.approx(total_time - 0.6 * DOOR_OPENING_TIME) == test_elevator.get_time_to_target()
   assert pytest.approx(test_elevator.get_doors_open()) == 0.4

   test_elevator.advance_simulation(total_time - DOOR_OPENING_TIME - DOOR_STAYING_OPEN_TIME)
   assert pytest.approx(0.4 * DOOR_OPENING_TIME + DOOR_STAYING_OPEN_TIME) == test_elevator.get_time_to_target()
   assert pytest.approx(test_elevator.get_doors_open()) == 0.6


   test_elevator.advance_simulation(0.1 * DOOR_OPENING_TIME)
   assert pytest.approx(0.3 * DOOR_OPENING_TIME + DOOR_STAYING_OPEN_TIME) == test_elevator.get_time_to_target()
   assert pytest.approx(test_elevator.get_doors_open()) == 0.7
