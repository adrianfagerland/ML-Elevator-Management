from elsim.elevator_simulator import ElevatorSimulator, DOOR_OPENING_TIME

def test_normal_arival_with_full_speed():
    test_elevator = ElevatorSimulator.Elevator(0,10,2,1)
    test_elevator.set_target_position(10)
    assert test_elevator.get_time_to_target() == 7

    test_elevator.door_opened_percentage = 1
    assert test_elevator.get_time_to_target() == 7 + DOOR_OPENING_TIME


def test_reverse_speed():
    test_elevator = ElevatorSimulator.Elevator(3,10,2,1,current_speed=-1)
    test_elevator.set_target_position(10)
    assert test_elevator.trajectory_list == [(3, -1, 0), 
                                             (2.5, 0, 1.0), 
                                             (4.5, 2, 2.0), 
                                             (8.0, 2, 1.75), 
                                             (10.0, 0, 2.0)]
    pass

def test_not_full_speed():
    test_elevator = ElevatorSimulator.Elevator(3 - 1/16,10,3,1,current_speed=1)
    #test_elevator.set_target_position(4)
    if(False):
        assert test_elevator.trajectory_list == [(3 - 1/16, 1, 0), 
                                                (3.25 - 1/32, 1.25, 0.25), 
                                                (4.0, 0.0, 1.25)]
    
    test_elevator = ElevatorSimulator.Elevator(1,10,6,2,current_speed=3)
    test_elevator.set_target_position(6)


    assert test_elevator.trajectory_list == [(1, 3, 0), 
                                             (2.375, 3.8078865529319543, 0.40394327646597716), 
                                             (6.0, 0.0, 1.9039432764659772)]
    
    pass
def test_overshot_speed():
    test_elevator = ElevatorSimulator.Elevator(3.5,10,2,1,current_speed=2)
    test_elevator.set_target_position(4)
    assert test_elevator.trajectory_list == [(3.5, 2, 0), (5.5, 0.0, 2.0), 
                                             (4.75, -1.224744871391589, 1.224744871391589), 
                                             (4.0, 0.0, 1.224744871391589)]
    pass