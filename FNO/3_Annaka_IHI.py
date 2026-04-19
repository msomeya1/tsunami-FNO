PATH_TO_MODEL_PARAMETER = "out/model.pt"


import os
import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F
from model_FNO import TsunamiFNO
from timeit import default_timer

def my_round(x):
    return int((2*x+1)//2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.device_count())

outdir = "IHI"
os.makedirs(outdir, exist_ok=True)




import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
import matplotlib.ticker as ticker


lon_min, lon_max = 131.2, 136.2
lat_min, lat_max = 29.5, 34.5
Nlon, Nlat = 251, 251
dlon, dlat = 0.02, 0.02

import matplotlib as mpl
cmap = mpl.colormaps.get_cmap("seismic")
cmap.set_bad(color='gray')



with open("../share/StationName_DONET2Nnet.txt") as f:
    Stations = [s.rstrip() for s in f.readlines()]

lonlat = np.loadtxt("../share/LonLat_DONET2Nnet.txt")
lon_st = lonlat[:,0]
lat_st = lonlat[:,1]
Nst = len(lon_st)

# M.MRA02, M.MRE19, M.MRG27, N.NAE05, N.NAE10, N.NBE04, N.NBE15
idx_selected = [1, 18, 26, 33, 38,  50, 61]

for idx in idx_selected:
    print(Stations[idx])


Nx = Nlon + 5
Ny = Nlat + 5
bathy = np.zeros((Nx, Ny), dtype=np.float32)
grd = xr.open_dataset("../share/Hyuganada.grd")
bathy0 = grd.z.data.reshape(Nlat, Nlon)

nanmask = np.ones_like(bathy0, dtype=float)
nanmask[bathy0 <= 0] = np.nan

bathy0 = np.rot90(bathy0, k=-1) # -> (Nlon, Nlat)
bathy0[bathy0<0.0] = 0.0
bathy[2:-3, 2:-3] = bathy0
bathy /= bathy.max()
bathy = torch.tensor(bathy)


padding = [
    2, 3, # 3rd dimension (latitude, size=Nlat) is padded (251->256)
    2, 3, # 2nd dimension (longitude, size=Nlon) is padded (251->256)
    0, 0  # 1st dimension (batch) is not padded
]






Nt = 32
width = 16
modes_x = 64 # less than Nlon
modes_y = 64 # less than Nlat
modes_t = 16 # less than Nt_out//2+1
FNO = TsunamiFNO(Nt, width, modes_x, modes_y, modes_t, bathy).to(device)
FNO.load_state_dict(torch.load(PATH_TO_MODEL_PARAMETER))
FNO = FNO.to(device)
FNO.eval()




# load data
obs_cache = {}

for ist in range(Nst):
    station = Stations[ist]
    lon = lon_st[ist]
    lat = lat_st[ist]
    i = my_round((lon-lon_min)/dlon)
    j = my_round((lat-lat_min)/dlat)

    eta = np.loadtxt(f"../data/Annaka/tgs-txt/{station}.txt")[::60]

    obs_cache[station] = (i, j, eta)


obs = np.zeros((Nlon, Nlat, Nt))
start_time = 1

for station in Stations:
    i, j, eta =obs_cache[station]
    obs[i, j, start_time-1:Nt] = eta[start_time:Nt+1]

amp = 2 * np.nanmax( np.abs(obs) )
sigma = np.nanmax( np.abs(obs) ) * 0.15 # model error of FNO (very rough). No obs error.
print(amp)


mask = np.where(
    obs == 0.0, 0, 1
)

obs = torch.from_numpy( 
    obs.astype(np.float32) 
).to(device)
mask = torch.from_numpy(
    mask.astype(np.float32) 
).to(device)


x0 = torch.nn.Parameter(torch.zeros(Nlon, Nlat, requires_grad=True, dtype=torch.float, device=device))

optimizer = torch.optim.LBFGS([x0])

def loss_fn(x):

    x = F.pad(x.reshape(1, Nlon, Nlat), padding) / amp
    out = FNO(x)[2:-3, 2:-3, :] * amp
    loss = (mask * (out - obs) ** 2).sum() / (2 * sigma ** 2)

    return loss


def closure():

    loss = loss_fn(x0)

    optimizer.zero_grad()
    loss.backward()

    return loss


epochs = 30

l = np.zeros(epochs)

t1 = default_timer()
for epoch in range(epochs):
    
    optimizer.step(closure)
    loss = closure()
    
    l[epoch] = loss.item()

    print(epoch, ": ", loss.item())

t2 = default_timer()
print(t2-t1)




fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
ax.semilogy(l, lw=2.5)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.grid(which='major', axis="x", linestyle=':', linewidth='0.5', color='black')
ax.grid(which='major', axis="y", linestyle=':', linewidth='0.5', color='black')
plt.savefig(f"{outdir}/loss_history.png", bbox_inches='tight')



def plot_array(array, fname):
    fig, ax = plt.subplots()
    im = ax.imshow(
        array, vmin=-1, vmax=+1, cmap=cmap, origin="lower",
        extent = [lon_min, lon_max, lat_min, lat_max]
    )

    ax.scatter(lon_st, lat_st, c="k")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(which="both", top=True, right=True)
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°N'))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Height [m]")

    plt.savefig(fname, dpi=400)


x_estimated = x0.to("cpu").detach().numpy().T 
print(np.abs(x_estimated).max())

plot_array(x_estimated * np.flipud(nanmask), f"{outdir}/eta0.png")


x = F.pad(x0.reshape(1, Nlon, Nlat), padding) / amp
out = FNO(x)[2:-3, 2:-3, :] * amp



ts1 = np.linspace(0, 32, 32+1)
ts2 = np.linspace(1, Nt, Nt)

for station in Stations:
    i, j, true = obs_cache[station]
    pred = out[i, j, :].to("cpu").detach().numpy()

    offset = max(true.max(), pred.max()) * 2

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ts1, true, label="obs")
    ax.plot(ts2, pred, label="inverted")
    ax.axvspan(start_time, Nt, color="gray", alpha=0.5)
    ax.legend()
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Height [m]")

    ax.set_title(station)
    plt.savefig(f"{outdir}/{station}.png")
    plt.close()




plt.rcParams["font.size"] = 16
fig, ax = plt.subplots(figsize=(6,8))

cnt = 0
for idx in reversed(idx_selected):
    station = Stations[idx]

    i, j, true = obs_cache[station]
    pred = out[i, j, :].to("cpu").detach().numpy()

    if cnt == 0:
        ax.plot(ts1, cnt + true, lw = 2, c = "black", label="Observation")
        ax.plot(ts2, cnt + pred, lw = 2, c = "red", label = "Inverted")
    else:
        ax.plot(ts1, cnt + true, lw = 2, c = "black")
        ax.plot(ts2, cnt + pred, lw = 2, c = "red")

    ax.text(34.5, cnt, station, va="center", fontsize=12)

    cnt += 1

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.axvspan(start_time, Nt, color="gray", alpha=0.5)

ax.set_xlabel("Time [min]")
ax.set_ylabel("Height [m]")
ax.minorticks_on()
plt.grid(which="major", color="gray")
ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right')

plt.tight_layout()

plt.savefig(f"{outdir}/waveforms_selected.png", dpi=400)


