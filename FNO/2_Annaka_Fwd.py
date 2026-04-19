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


outdir = "Annaka_Fwd"
os.makedirs(outdir, exist_ok=True)

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as patches


lon_min, lon_max = 131.2, 136.2
lat_min, lat_max = 29.5, 34.5
Nlon, Nlat = 251, 251
dlon, dlat = 0.02, 0.02

import matplotlib as mpl
cmap = mpl.colormaps.get_cmap("seismic")
cmap.set_bad(color='gray')


with open("../../share/StationName_DONET2Nnet.txt") as f:
    Stations = [s.rstrip() for s in f.readlines()]

lonlat = np.loadtxt("../../share/LonLat_DONET2Nnet.txt")
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
grd = xr.open_dataset("../../share/Hyuganada.grd")
bathy0 = grd.z.data.reshape(Nlat, Nlon)

nanmask = np.ones_like(bathy0, dtype=float)
nanmask[bathy0 <= 0] = np.nan

bathy0 = np.rot90(bathy0, k=-1) # -> (Nlon, Nlat)
bathy0[bathy0<0.0] = 0.0
bathy[2:-3, 2:-3] = bathy0
bathy /= bathy.max()
bathy = torch.tensor(bathy)

grd_true = xr.open_dataset(f"../../data/Annaka/SD01.nc")
eta0_true = np.nan_to_num(grd_true.wave_height.data[0,:,:])
eta8_true = np.nan_to_num(grd_true.wave_height.data[8,:,:])
eta16_true = np.nan_to_num(grd_true.wave_height.data[16,:,:])
eta32_true = np.nan_to_num(grd_true.wave_height.data[32,:,:])

amp = np.abs(eta0_true).max() * 2
print(amp)


Nt = 32
width = 16
modes_x = 64 # less than Nlon
modes_y = 64 # less than Nlat
modes_t = 16 # less than Nt_out//2+1
FNO = TsunamiFNO(Nt, width, modes_x, modes_y, modes_t, bathy).to(device)
FNO.load_state_dict(torch.load(PATH_TO_MODEL_PARAMETER))
FNO = FNO.to(device)
FNO.eval()




padding = [
    2, 3, # 3rd dimension (latitude, size=Nlat) is padded (251->256)
    2, 3, # 2nd dimension (longitude, size=Nlon) is padded (251->256)
    0, 0  # 1st dimension (batch) is not padded
]

x = torch.from_numpy(eta0_true.T.astype(np.float32)).to(device) / amp
x = F.pad(x.reshape(1, Nlon, Nlat), padding) 
out = FNO(x)[2:-3, 2:-3, :] * amp


eta8_pred = out[:, :, 8-1].T.to("cpu").detach().numpy()
eta16_pred = out[:, :, 16-1].T.to("cpu").detach().numpy()
eta32_pred = out[:, :, 32-1].T.to("cpu").detach().numpy()

import pandas as pd
import pyproj


lon_mid, lat_mid = (lon_min+lon_max)/2, (lat_min+lat_max)/2
ll2xy = pyproj.Transformer.from_crs(
    crs_from="EPSG:4326", # WGS84
    crs_to=f"+proj=tmerc +lon_0={lon_mid} +lat_0={lat_mid} +ellps=WGS84 +datum=WGS84 +units=km", 
    always_xy=True
)


df = pd.read_csv(
    f"../../data/Annaka/fault_param.txt", comment="!", sep="\s+",
    names = ("lat", "lon", "depth", "length", "width", "dip", "strike", "rake", "slip")
)
lon, lat, length, width, dip, strike = \
    df["lon"].item(), df["lat"].item(), df["length"].item(), df["width"].item(), df["dip"].item(), df["strike"].item()

ss = np.sin(np.deg2rad(strike))
cs = np.cos(np.deg2rad(strike))
sd = np.sin(np.deg2rad(dip))
cd = np.cos(np.deg2rad(dip))

x, y = ll2xy.transform(lon, lat) 

x_corners = [
    x,
    x               + width * cd * cs,
    x + length * ss + width * cd * cs,
    x + length * ss
]
y_corners = [
    y,
    y               - width * cd * ss,
    y + length * cs - width * cd * ss,
    y + length * cs
]

lon_corners, lat_corners = ll2xy.transform(x_corners, y_corners, direction = "INVERSE")
corners = np.column_stack((np.array(lon_corners), np.array(lat_corners)))


def plot_array(array, fname, rect = False):
    fig, ax = plt.subplots()
    im = ax.imshow(
        array, vmin=-1, vmax=+1, cmap=cmap, origin="lower",
        extent = [lon_min, lon_max, lat_min, lat_max]
    )

    if rect:
        rect = patches.Polygon(corners, ls="--", lw=1.5, closed=True, edgecolor='dimgray', fill=False)
        ax.add_patch(rect)
        ax.plot(    
            [corners[0, 0], corners[3, 0]],
            [corners[0, 1], corners[3, 1]],
            color='dimgray', lw=1.5
        )

    for idx in idx_selected:
        ax.scatter(lon_st[idx], lat_st[idx], c="k")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(which="both", top=True, right=True)
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°N'))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Height [m]")

    plt.savefig(fname, dpi=400)



plot_array(eta0_true * np.flipud(nanmask), f"{outdir}/N4_true0.png", rect=True)
plot_array(eta8_true * np.flipud(nanmask), f"{outdir}/N4_true8.png")
plot_array(eta16_true * np.flipud(nanmask), f"{outdir}/N4_true16.png")
plot_array(eta32_true * np.flipud(nanmask), f"{outdir}/N4_true32.png")

plot_array(eta8_pred * np.flipud(nanmask), f"{outdir}/N4_pred8.png")
plot_array(eta16_pred * np.flipud(nanmask), f"{outdir}/N4_pred16.png")
plot_array(eta32_pred * np.flipud(nanmask), f"{outdir}/N4_pred32.png")

plot_array((eta8_pred-eta8_true) * np.flipud(nanmask), f"{outdir}/N4_diff8.png")
plot_array((eta16_pred-eta16_true) * np.flipud(nanmask), f"{outdir}/N4_diff16.png")
plot_array((eta32_pred-eta32_true) * np.flipud(nanmask), f"{outdir}/N4_diff32.png")





win1d = np.hanning(Nlon)
win2d = np.outer(win1d, win1d)

def amp_spectrum(array):
    fft = np.fft.fft2(array*win2d)
    shifted = np.fft.fftshift(fft)
    return np.abs(shifted)


eta0_true_spectrum = amp_spectrum(eta0_true)
eta16_true_spectrum = amp_spectrum(eta16_true)
eta32_true_spectrum = amp_spectrum(eta32_true)

eta16_pred_spectrum = amp_spectrum(eta16_pred)
eta32_pred_spectrum = amp_spectrum(eta32_pred)


def plot_spectrum(array, cmap, fname, kmax_grid=False):
    fig, ax = plt.subplots()
    im = ax.imshow(
        array, cmap=cmap, norm=colors.LogNorm(vmin=0.1, vmax=10),
        extent = [-Nlon//2, Nlon//2, -Nlat//2, Nlat//2]
    )

    rect = np.array([
        [-64, -64],
        [-64, +64],
        [+64, +64],
        [+64, -64],
        [-64, -64],
    ])
    x, y = rect[:, 0], rect[:, 1]
    ax.plot(x, y, "k--")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude Spectrum")

    plt.tight_layout()
    plt.savefig(fname, dpi=400)


plot_spectrum(eta32_true_spectrum, "jet", f"{outdir}/N4_spectrum_true32.png")
plot_spectrum(eta32_pred_spectrum, "jet", f"{outdir}/N4_spectrum_pred32.png")
plot_spectrum(eta32_pred_spectrum / eta32_true_spectrum, "seismic", f"{outdir}/N4_spectrum_diff32.png", kmax_grid=True)


ts1 = np.linspace(0, 32, 32*60+1)
ts2 = np.linspace(1, 32, 32)

for ist in range(Nst):
    station = Stations[ist]

    true = np.loadtxt(f"../../data/Annaka/tgs-txt/{station}.txt")

    lon = lon_st[ist]
    lat = lat_st[ist]
    i = my_round((lon-lon_min)/dlon)
    j = my_round((lat-lat_min)/dlat)
    pred = out[i, j, :].to("cpu").detach().numpy()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ts1, true, label="true")
    ax.plot(ts2, pred, label="pred")
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

    true = np.loadtxt(f"../../data/Annaka/tgs-txt/{station}.txt")
    lon = lon_st[idx]
    lat = lat_st[idx]
    i = my_round((lon-lon_min)/dlon)
    j = my_round((lat-lat_min)/dlat)
    pred = out[i, j, :].to("cpu").detach().numpy()

    if cnt == 0:
        ax.plot(ts1, cnt + true, lw = 2, c = "black", label="Ground Truth")
        ax.plot(ts2, cnt + pred, lw = 2, c = "red", label = "FNO Prediction")
    else:
        ax.plot(ts1, cnt + true, lw = 2, c = "black")
        ax.plot(ts2, cnt + pred, lw = 2, c = "red")

    ax.text(34.5, cnt, station, va="center", fontsize=12)

    cnt += 1

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.set_xlabel("Time [min]")
ax.set_ylabel("Height [m]")
ax.minorticks_on()
plt.grid(which="major", color="gray")
ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right')

plt.tight_layout()

plt.savefig(f"{outdir}/N4_forward_selected.png", dpi=400)


