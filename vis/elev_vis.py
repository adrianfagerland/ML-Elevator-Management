def print_elevator(elev_positions, buttons_out, buttons_in, elev_speed):

    i = len(buttons_out)
    for k in range(0, len(buttons_out)):
        i = i - 1
        print(i,"|\t", end="")
        for e_it, e in enumerate(elev_positions):

            if e == i:
                elev_dir = 0
                if elev_speed[e_it] != 0:
                    elev_dir = elev_speed[e_it] / abs(elev_speed[e_it]) # 1, 0, -1

                if elev_dir == 1:
                    print("\t^", end="")
                elif elev_dir == -1:
                    print("\tv", end="")
                else:
                    print("\to", end="")

            else:
                print("\t.", end="")
        print("\t|", end="")
        if buttons_out[i]["up"] == True:
            print("^", end = "")
        if buttons_out[i]["down"] == True:
            print("v", end = "")
        print()

    for e_it, e in enumerate(elev_positions):
        print("Elev", e_it,": ", end="")
        for f_it, f in enumerate(buttons_in[e_it]):
            if f == True:
                print(f_it, "", end="")
        print()