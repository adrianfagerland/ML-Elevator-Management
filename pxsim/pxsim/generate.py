import argparse
import datetime
import random
from typing import Generator, Tuple

from pxsim.distribution import Distribution

NUM_WEEKS = 52
NUM_FLOORS = 10
START_TIME = datetime.datetime(2002, 12, 30, 0, 0, 0)

MIN_ENTRIES_PER_DAY = 700
MAX_ENTRIES_PER_DAY = 1000
WEEKEND_FACTOR = 0.8


def generate_weeks(num_weeks: int = NUM_WEEKS, num_floors: int = NUM_FLOORS, start_time: datetime.datetime = START_TIME) -> Generator[Tuple[datetime.datetime, int, int], None, None]:
    """A generator that yields tuples of the form (timestamp, startfloor, endfloor) for a given number of weeks

    Args:
        num_weeks (int, optional): How many weeks shall the generator yield for. Defaults to NUM_WEEKS.
        num_floors (int, optional): The number of floors in the simulated building. Defaults to NUM_FLOORS.
        start_time (datetime.datetime, optional): The simulation will run from the start of the week that `start_time` is a part of. Defaults to START_TIME.

    Yields:
        Generator[Tuple, None, None]: Generates tuples of the form (timestamp, startfloor, endfloor)
    """
    distribution = Distribution()
    start_of_week = start_time.date() - datetime.timedelta(days=start_time.weekday())
    start_of_week_datetime = datetime.datetime.combine(
        start_of_week, datetime.time.min)
    for _ in range(num_weeks):
        for entry in generate_entries_for_week(num_floors, start_of_week_datetime, distribution):
            yield entry
        start_of_week_datetime = start_of_week_datetime + \
            datetime.timedelta(days=7)


def generate_entries_for_week(num_floors: int, start_time: datetime.datetime, distribution: Distribution) -> Generator[Tuple[datetime.datetime, int, int], None, None]:
    for day in range(7):
        start_of_day = start_time + datetime.timedelta(days=day)
        if day < 5:
            weekend = False
        else:
            weekend = True

        num_of_entries = int(random.randint(
            MIN_ENTRIES_PER_DAY, MAX_ENTRIES_PER_DAY) * (WEEKEND_FACTOR if weekend else 1))
        entries = distribution.generate_distribution(num_of_entries)
        entries.sort()
        for second in entries:
            startfloor = random.randint(0, num_floors - 1)
            endfloor = random.randint(0, num_floors - 1)
            while endfloor == startfloor:
                endfloor = random.randint(0, num_floors - 1)
            yield (start_of_day + datetime.timedelta(seconds=second), startfloor, endfloor)


def write_to_csv(num_weeks, num_floors):
    """
    Generates a csv file where data from `generate_weeks()` is written to file
    """
    from poetry_version import extract
    with open(f'../pxsim/data/w{num_weeks}_f{num_floors}_{extract(__file__)}.csv', 'w') as f:
        f.write('timestamp,startfloor,endfloor\n')
        for entry in generate_weeks(num_weeks, num_floors):
            f.write(f'{entry[0].isoformat()}, {entry[1]}, {entry[2]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate tuples of the form (timestamp, startfloor, endfloor) for a given number of weeks')
    parser.add_argument('num_weeks', type=int,
                        help='How many weeks shall the generator yield for')
    parser.add_argument('num_floors', type=int,
                        help='The number of floors in the simulated building')
    args = parser.parse_args()

    write_to_csv(args.num_weeks, args.num_floors)
