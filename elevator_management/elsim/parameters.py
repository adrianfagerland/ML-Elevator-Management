from typing import NewType

# All time values in seconds
DOOR_OPENING_TIME = 2
DOOR_STAYING_OPEN_TIME = 3

FLOOR = NewType("FLOOR", int)
TIME_SEC = NewType("TIME_SEC", float)

DIST_EPSILON = 0.01  # 1/100th of an floor allowed error due to math inaccuracys
LOSS_FACTOR = 1e6