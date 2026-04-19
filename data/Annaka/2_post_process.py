import os
import numpy as np
import matplotlib.pyplot as plt



def read_tgsfile(ist):
    ist = str(ist).zfill(6)
    file_path = f"tgsfiles/tgs{ist}"
    eta = np.array([])

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if "step=" in line:
                eta = np.append(eta, float(parts[4]))

    return eta


os.makedirs("tgs-png", exist_ok=True)
os.makedirs("tgs-txt", exist_ok=True)


dt = 1 # sec
tmin = 0
tmax = 32 * 60 # 32 min. = 1920 sec.
Nt = int(tmax/dt) + 1
ts = np.linspace(tmin, tmax, Nt)


with open("../../share/StationName_DONET2Nnet.txt") as f:
    Stations = [s.rstrip() for s in f.readlines()]

Nst = len(Stations)



for ist, station in enumerate(Stations):
    eta = read_tgsfile(ist+1)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ts/60, eta)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Height [m]")
    ax.set_title(station)
    plt.savefig(f"tgs-png/{station}.png")
    plt.close()

    np.savetxt(f"tgs-txt/{station}.txt", eta)