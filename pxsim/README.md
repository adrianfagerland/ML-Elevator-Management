# Simulating people input

People show up to the elevator, they already know where they are going.
When they arrive, they either press up or down.

Generating data for a week as a `.csv` on the format:
timestamp (%Y-%m-%d %H:%M:%S), startfloor, endfloor

`python3 pxsim/generate.py --num-days 7 --num-entries-per-day 1000 --num-floors 10 --start-time "2022-01-01 08:00:00" --end-time "2022-01-07 23:59:59" --output-file "elevator_data.csv"`
