def print_elevator(observations, skipped_time, previous_action=None, setup=False):
    elev_positions = observations["position"]
    buttons_out = observations["floors"]
    buttons_in = observations["buttons"]
    elev_speed = observations["speed"]
    px = observations["elevators_occupancy"]
    doors_state = observations["doors_state"]
    doors_moving_direction = observations["doors_moving_direction"]
    i = len(buttons_out)
    print_str = ""
    for _ in range(len(buttons_out)):
        i = i - 1
        # make sure that the length of i printed is the same
        # even if it is 1 or 2 digits
        print_str += f"{i:2}|\t"
        for e_it, e in enumerate(elev_positions):
            if round(e) == i:
                elev_dir = 0
                if elev_speed[e_it] != 0:
                    elev_dir = elev_speed[e_it] / abs(elev_speed[e_it])  # 1, 0, -1

                door_state = round(doors_state[e_it], 8)
                if elev_dir == 1:
                    print_str += "\t^"
                elif elev_dir == -1:
                    print_str += "\tv"
                elif door_state == 0:
                    print_str += "\tx"
                elif door_state == 1:
                    print_str += "\to"
                    if previous_action is None:
                        continue
                    if previous_action["next_move"][e_it] == 1:
                        print_str += "^"
                    elif previous_action["next_move"][e_it] == -1:
                        print_str += "v"
                elif doors_moving_direction[e_it] == -1:
                    print_str += f"    {' '*(e_it == 0)}  >o<"
                elif doors_moving_direction[e_it] == 1:
                    print_str += f"   {' '*(e_it == 0)}   <o>"
                else:
                    raise Exception("Unexpected door state")

            else:
                print_str += "\t."
        print_str += "\t|"
        end_str = ""
        if buttons_out[i][0] == True:
            end_str += "^"
        if buttons_out[i][1] == True:
            end_str += "v"
        end_str += " " * (4 - len(end_str))
        print_str += end_str + "\n"

    for e_it, e in enumerate(elev_positions):
        print_str += "Elev " + str(e_it) + ": "
        for f_it, f in enumerate(buttons_in[e_it]):
            board_str = ""
            if f == True:
                board_str += str(f_it)
            print_str += " " * (3 - len(board_str)) + board_str
        print_str += "Souls on board:" + str(px[e_it]) + "\n"

    if not setup:
        num_lines = len(buttons_out) + len(buttons_in)
        line_length = max([len(line) for line in print_str.splitlines()])
        for _ in range(num_lines):
            delete_last_line(line_length)
    print(print_str, end="")


def delete_last_line(line_length):
    print("\033[A" + " " * (line_length + 3) + "\033[A")
