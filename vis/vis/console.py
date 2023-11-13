from time import sleep


def print_elevator(elev_positions, buttons_out, buttons_in, elev_speed, px, setup=False):
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
        print_str += (f"Elev {e_it}: ")
        for f_it, f in enumerate(buttons_in[e_it]):
            if f == True:
                print_str += str(f_it)
        print_str += "Souls on board:" + str(px[e_it]) + "\n"

    num_lines = len(print_str.splitlines()) + 1
    if (not setup):
        for _ in range(num_lines):
            delete_last_line()
    print(print_str)


def delete_last_line():
    print("\033[A                             \033[A")
