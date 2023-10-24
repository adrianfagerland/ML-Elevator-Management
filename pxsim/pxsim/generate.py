import argparse
import csv
import datetime
import random

DEFAULT_NUM_DAYS = 7
DEFAULT_NUM_ENTRIES_PER_DAY = 1000
DEFAULT_NUM_FLOORS = 10
DEFAULT_START_TIME = datetime.datetime(2022, 1, 1, 8, 0, 0)
DEFAULT_END_TIME = datetime.datetime(2022, 1, 7, 23, 59, 59)
DEFAULT_OUTPUT_FILE = "elevator_data.csv"

parser = argparse.ArgumentParser(description="Generate simulated elevator data")

parser.add_argument("--num-days", type=int, default=DEFAULT_NUM_DAYS, help="number of days to simulate")
parser.add_argument("--num-entries-per-day", type=int, default=DEFAULT_NUM_ENTRIES_PER_DAY, help="number of entries per day")
parser.add_argument("--num-floors", type=int, default=DEFAULT_NUM_FLOORS, help="number of floors in the building")
parser.add_argument("--start-time", type=str, default=DEFAULT_START_TIME.strftime("%Y-%m-%d %H:%M:%S"), help="start time for the simulation (format: YYYY-MM-DD HH:MM:SS)")
parser.add_argument("--end-time", type=str, default=DEFAULT_END_TIME.strftime("%Y-%m-%d %H:%M:%S"), help="end time for the simulation (format: YYYY-MM-DD HH:MM:SS)")
parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="output file name")

args = parser.parse_args()

start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")

with open(args.output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["timestamp", "startfloor", "endfloor"])

    for day in range(args.num_days):
        for entry in range(args.num_entries_per_day):
            timestamp = start_time + datetime.timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))

            startfloor = random.randint(1, args.num_floors)
            endfloor = random.randint(1, args.num_floors)

            while startfloor == endfloor:
                endfloor = random.randint(1, args.num_floors)

            writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), startfloor, endfloor])

print(f"Generated {args.num_days * args.num_entries_per_day} entries in {args.output_file}")
