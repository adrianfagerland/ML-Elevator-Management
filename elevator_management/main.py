from ml import NearestCar, RandomScheduler, Runner

runner = Runner(
    NearestCar, num_elevators=3, num_floors=10, max_speed=1, max_acceleration=1, seed=0, visualizer="pygame"
)

runner.run(visualize=True)
