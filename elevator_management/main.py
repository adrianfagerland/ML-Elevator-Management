from ml import NearestCar, Runner

runner = Runner(
    NearestCar,
    num_elevators=12,
    num_floors=10,
    max_speed=1,
    max_acceleration=1,
    seed=0,
    should_plot=True,
)

runner.run(visualize=True)
