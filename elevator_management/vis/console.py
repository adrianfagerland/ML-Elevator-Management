from time import sleep


# def print_elevator(elev_positions, buttons_out, buttons_in, elev_speed, px, setup=False):
def print_elevator(observations, previous_door_state, setup=False):
    elev_positions = observations['position']
    buttons_out = observations['floors']
    buttons_in = observations['buttons']
    elev_speed = observations['speed']
    px = observations['elevators_occupancy']
    doors_state = observations['doors_state']

    i = len(buttons_out)
    print_str = "\n\nSimulation:" + " " * 30 + " \n"
    for k in range(len(buttons_out)):
        i = i - 1
        # make sure that the length of i printed is the same
        # even if it is 1 or 2 digits
        print_str += f"{i:2}|\t"
        for e_it, e in enumerate(elev_positions):

            if round(e) == i:
                elev_dir = 0
                if elev_speed[e_it] != 0:
                    elev_dir = elev_speed[e_it] / abs(elev_speed[e_it])  # 1, 0, -1

                if elev_dir == 1:
                    print_str += ("\t^")
                elif elev_dir == -1:
                    print_str += ("\tv")
                elif doors_state[e_it] < previous_door_state[e_it]:
                    print_str += (f"    {' '*(e_it == 0)}  >o<")
                elif doors_state[e_it] > previous_door_state[e_it]:
                    print_str += (f"   {' '*(e_it == 0)}   <o>")
                elif doors_state[e_it] == 0:
                    print_str += ("\tx")
                else:
                    print_str += ("\to")

            else:
                print_str += ("\t.")
        print_str += ("\t|")
        end_str = ""
        if buttons_out[i][0] == True:
            end_str += ("^")
        if buttons_out[i][1] == True:
            end_str += ("v")
        end_str += " " * (4-len(end_str))
        print_str += end_str + "\n"

    for e_it, e in enumerate(elev_positions):
        print_str += "Elev " + str(e_it) + ": "
        for f_it, f in enumerate(buttons_in[e_it]):
            board_str = ""
            if f == True:
                board_str += str(f_it)
            print_str += " " * (3 - len(board_str)) + board_str
        print_str += "Souls on board:" + str(px[e_it]) + "\n"

    if (not setup):
        num_lines = len(print_str.splitlines()) + 1
        line_length = max([len(line) for line in print_str.splitlines()])
        for _ in range(num_lines):
            delete_last_line(line_length)
    print(print_str)


def delete_last_line(line_length):
    print("\033[A" + " " * (line_length + 3) + "\033[A")
