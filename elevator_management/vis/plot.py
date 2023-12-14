import sys

import pygame

# difine colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (240, 240, 240)
DARK_GRAY = (150, 150, 150)
LIGHT = (242, 200, 92)

# difine positions
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

ELEVATOR_WIDTH = 20
ELEVATOR_HEIGHT = 30
X_GAP = 20
GROUND_FLOOR = 500
# X_OFFSET = 250
Y_OFFSET = 10
FLOOR_SIZE = ELEVATOR_HEIGHT + Y_OFFSET
X_OFFSET = (
    lambda num_elev: (WINDOW_WIDTH / 2) - num_elev * (ELEVATOR_WIDTH + X_GAP) * 0.5 - 10
)


class Visualizer:
    def __init__(self, observations, width=WINDOW_WIDTH, height=WINDOW_HEIGHT) -> None:
        pygame.init()

        # create window
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Tsinghua Elevator")

        # get bsic parameter
        elev_positions = observations["position"]
        buttons_out = observations["floors"]
        self.num_elev = len(elev_positions)
        self.num_floors = len(buttons_out)

        # create elevator sprites
        self.elevators_plots = []
        self.all_sprites = pygame.sprite.Group()  # all_sprites maybe not needed

        for idx, e in enumerate(elev_positions):
            elevator: Elevator = Elevator(idx, observations, self.window)
            self.elevators_plots.append(elevator)

    def update(self, observations, action):
        # draw all. The order matters

        self.window.fill(WHITE)
        self.draw_surrounding(observations)

        for e in self.elevators_plots:
            e.update_observation(observations)

        self.all_sprites.update(observations)

        for e in self.elevators_plots:
            if action is None:
                continue
            if (
                observations["doors_state"][e.number] == 0
                and action["target"][e.number] != observations["position"][e.number]
            ):
                font = pygame.font.Font("freesansbold.ttf", 12)
                t = int(action["target"][e.number])
                text = font.render(str(t), True, LIGHT)
                self.window.blit(
                    text,
                    (
                        e.body.rect.x + ELEVATOR_WIDTH * 0.5 - 3,
                        e.body.rect.y + ELEVATOR_HEIGHT * 0.5 - 6,
                    ),
                )

        pygame.display.flip()

    def draw_surrounding(self, observations):
        # draw sky
        pygame.draw.rect(
            self.window, (214, 239, 255), (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        # draw gras
        pygame.draw.polygon(
            self.window,
            (97, 184, 118),
            [
                (0, WINDOW_HEIGHT),
                (0, GROUND_FLOOR),
                (WINDOW_WIDTH, GROUND_FLOOR),
                (WINDOW_WIDTH, WINDOW_HEIGHT),
            ],
        )
        # draw sun
        pygame.draw.circle(self.window, (255, 252, 99), (WINDOW_WIDTH - 50, 50), 100)

        # draw house

        house_x = X_OFFSET(self.num_elev) - (0.5 * X_GAP)
        house_w = self.num_elev * (ELEVATOR_WIDTH + X_GAP)
        house_h = FLOOR_SIZE * (self.num_floors)
        house_y = GROUND_FLOOR - house_h

        pygame.draw.rect(
            self.window, GRAY, (house_x, house_y, house_w, house_h)
        )  # draw border of house

        # print floor numbers
        font = pygame.font.Font("freesansbold.ttf", 12)
        for floor_num in range(0, self.num_floors):
            text = font.render(str(floor_num), True, BLACK)
            self.window.blit(
                text,
                (
                    house_x - 15,
                    (GROUND_FLOOR - floor_num * FLOOR_SIZE) - (FLOOR_SIZE / 2),
                ),
            )

            k = GROUND_FLOOR - floor_num * FLOOR_SIZE
            pygame.draw.line(
                self.window, BLACK, (house_x, k), (house_x + house_w - 1, k), 1
            )

        # print calls
        calls = observations["floors"]
        DISTANCE_BETWEEN_BUTTONS = 35
        for floor_number, call in enumerate(calls):
            a = house_x + house_w + 15
            b = GROUND_FLOOR - floor_number * FLOOR_SIZE - 8
            g = 20
            h = 15

            # up pointing button
            c1 = DARK_GRAY
            if call[0] == 1:
                c1 = LIGHT
            pygame.draw.polygon(
                self.window, c1, [(a, b), (a + g, b), (a + (g / 2), b - h)]
            )

            # down pointing button
            c2 = DARK_GRAY
            if call[1] == 1:
                c2 = LIGHT
            pygame.draw.polygon(
                self.window,
                c2,
                [
                    (a + DISTANCE_BETWEEN_BUTTONS, b - h),
                    (a + DISTANCE_BETWEEN_BUTTONS + g, b - h),
                    (a + DISTANCE_BETWEEN_BUTTONS + (g / 2), b),
                ],
            )

        pygame.draw.rect(
            self.window, DARK_GRAY, (house_x, house_y, house_w, house_h), 2
        )  # 2 is the border thickness


class Elevator:
    # Elevator contains a Sprite Group made out of Body and Door
    def __init__(self, number, observation, window):
        super(Elevator, self).__init__()
        self.number = number
        self.num_elevators = len(observation["position"])
        self.num_floors = len(observation["floors"])

        self.window = window
        self.group = pygame.sprite.Group()

        # ELEVATO
        self.body = ElevatorBody(number, observation)
        self.group.add(self.body)
        # DOORS
        self.door = ElevatorDoor(number, observation, window)
        self.group.add(self.door)

    def update_observation(self, observation):
        # print cable

        pygame.draw.line(
            self.window,
            DARK_GRAY,
            (self.body.rect.x + (ELEVATOR_WIDTH * 0.5), self.body.rect.y),
            (
                self.body.rect.x + (ELEVATOR_WIDTH * 0.5),
                GROUND_FLOOR - self.num_floors * FLOOR_SIZE,
            ),
            2,
        )

        # self.door.update(observation)
        self.group.update(observation)

        self.group.draw(self.window)


class ElevatorBody(pygame.sprite.Sprite):
    def __init__(self, number, observation):
        super(ElevatorBody, self).__init__()

        self.number = number
        self.y = observation["position"][self.number]

        x = number * (ELEVATOR_WIDTH + X_GAP) + X_OFFSET(len(observation["position"]))
        y = GROUND_FLOOR - FLOOR_SIZE * self.y

        self.image = pygame.Surface((ELEVATOR_WIDTH, ELEVATOR_HEIGHT))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self, observation):
        x = self.number * (ELEVATOR_WIDTH + X_GAP) + X_OFFSET(
            len(observation["position"])
        )
        y = (
            GROUND_FLOOR
            - FLOOR_SIZE * observation["position"][self.number]
            - ELEVATOR_HEIGHT
        )
        self.rect.y = y


DOOR_WIDTH = ELEVATOR_WIDTH - 6
DOOR_HEIGHT = ELEVATOR_HEIGHT - 6


class ElevatorDoor(pygame.sprite.Sprite):
    def __init__(self, number, observation, window):
        super(ElevatorDoor, self).__init__()
        self.number = number
        self.y = observation["position"][self.number]
        self.window = window

        body_x = number * (ELEVATOR_WIDTH + X_GAP) + X_OFFSET(
            len(observation["position"])
        )
        body_y = GROUND_FLOOR - FLOOR_SIZE * self.y

        self.image = pygame.Surface((DOOR_WIDTH, DOOR_HEIGHT))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()

    def update(self, observation):
        self.y = observation["position"][self.number]
        doors_state = observation["doors_state"]
        body_x = self.number * (ELEVATOR_WIDTH + X_GAP) + X_OFFSET(
            len(observation["position"])
        )
        body_y = GROUND_FLOOR - FLOOR_SIZE * self.y
        c = doors_state[self.number] * 255
        self.image.fill((c, c, c))

        self.rect.x = body_x + (ELEVATOR_WIDTH - DOOR_WIDTH) / 2
        self.rect.y = body_y - DOOR_HEIGHT - (ELEVATOR_HEIGHT - DOOR_HEIGHT) / 2
