import datetime
from typing import Generator, Tuple

from pxsim.building import Building

START_TIME = datetime.datetime(2002, 12, 30, 0, 0, 0)
SEED = 42


def generate_arrivals(
    num_floors: int,
    num_elevators: int,
    density: float,
    num_arrivals: int,
    seed: int = SEED,
    start_time: datetime.datetime = START_TIME,
) -> Generator[Tuple[datetime.datetime, int, int], None, None]:
    """A generator that yields tuples of the form (timestamp, startfloor, endfloor).

    Args:
        num_floors (int): The number of floors in the simulated building.
        num_elevators (int): The number of elevators in the simulated building.
        density (float): The factor applied to the default number of arrivals.
        num_arrivals (int): The number of arrivals to simulate.

    Yields:
        Generator[Tuple[datetime.datetime, int, int], None, None]: A generator of the arrivals on the form (timestamp, startfloor, endfloor).
    """
    current_time = start_time
    building = Building(num_floors, num_elevators, density, seed=seed)
    for _ in range(num_arrivals):
        current_time, arrivals = building.get_next_arrivals(current_time)
        for arrival in arrivals:
            yield arrival


# def write_to_csv(num_weeks, num_floors):
#     """
#     Generates a csv file where data from `generate_weeks()` is written to file
#     """
#     import importlib.metadata
#     with open(f'../pxsim/data/w{num_weeks}_f{num_floors}_{importlib.metadata.version("pxsim")}.csv', 'w') as f:
#         f.write('timestamp,startfloor,endfloor\n')
#         # for entry in generate_weeks(num_weeks, num_floors):
#         #     f.write(f'{entry[0].isoformat()}, {entry[1]}, {entry[2]}\n')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Generate tuples of the form (timestamp, startfloor, endfloor) for a given number of weeks')
#     parser.add_argument('num_weeks', type=int,
#                         help='How many weeks shall the generator yield for')
#     parser.add_argument('num_floors', type=int,
#                         help='The number of floors in the simulated building')
#     args = parser.parse_args()

#     write_to_csv(args.num_weeks, args.num_floors)
