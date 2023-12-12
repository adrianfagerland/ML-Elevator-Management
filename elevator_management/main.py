from ml import NearestCar, Runner

runner = Runner(
    NearestCar, num_elevators=3, num_floors=10, max_speed=1, max_acceleration=1, seed=0
)

runner.run(visualize=True)
