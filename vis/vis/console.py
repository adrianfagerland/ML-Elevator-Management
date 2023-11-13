def print_elevator(elev_positions, buttons_out, buttons_in, elev_speed, px):
    i = len(buttons_out)
    for k in range(len(buttons_out)):
        i = i - 1
        # make sure that the length of i printed is the same
        # even if it is 1 or 2 digits
        print(f"{i:2}", "|\t", end="")
        for e_it, e in enumerate(elev_positions):

            if e == i:
                elev_dir = 0
                if elev_speed[e_it] != 0:
                    elev_dir = elev_speed[e_it] / abs(elev_speed[e_it])  # 1, 0, -1

                if elev_dir == 1:
                    print("\t^", end="")
                elif elev_dir == -1:
                    print("\tv", end="")
                else:
                    print("\to", end="")

            else:
                print("\t.", end="")
        print("\t|", end="")
        if buttons_out[i][0] == True:
            print("^", end="")
        if buttons_out[i][1] == True:
            print("v", end="")
        print()

    for e_it, e in enumerate(elev_positions):
        print("Elev", e_it, ": ", end="")
        for f_it, f in enumerate(buttons_in[e_it]):
            if f == True:
                print(f_it, "", end="")
        print("Souls on board:", px[e_it])
        print()
