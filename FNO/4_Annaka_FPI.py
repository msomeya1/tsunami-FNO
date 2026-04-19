PATH_TO_MODEL_PARAMETER = "out/model.pt"


import os
import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F
from model_FNO import TsunamiFNO
import pyproj
from OkadaTorch import OkadaWrapper
okada = OkadaWrapper()
from timeit import default_timer

def my_round(x):
    return int((2*x+1)//2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.device_count())

outdir = "syn2"
os.makedirs(outdir, exist_ok=True)


import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
import matplotlib.ticker as ticker
import matplotlib.patches as patches

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
    i, j, eta = obs_cache[station]
    obs[i, j, start_time-1:Nt] = eta[start_time:Nt+1]


sigma = np.nanmax( np.abs(obs) ) * 0.15 # model error of FNO (very rough). No obs error.
print(sigma)

mask = np.where(
    obs == 0.0, 0, 1
)

obs = torch.from_numpy( 
    obs.astype(np.float32) 
).to(device)
mask = torch.from_numpy(
    mask.astype(np.float32) 
).to(device)


# coordinate transformation
lon_mid, lat_mid = (lon_min+lon_max)/2, (lat_min+lat_max)/2
ll2xy = pyproj.Transformer.from_crs(
    crs_from="EPSG:4326", # WGS84
    crs_to=f"+proj=tmerc +lon_0={lon_mid} +lat_0={lat_mid} +ellps=WGS84 +datum=WGS84 +units=km", 
    always_xy=True
)

lon = np.linspace(lon_min, lon_max, Nlon)
lat = np.linspace(lat_min, lat_max, Nlat)
Lon, Lat = np.meshgrid(lon, lat)
X, Y = ll2xy.transform(Lon, Lat)

coords = {
    "x": torch.from_numpy(X).to(torch.float32).to(device),
    "y": torch.from_numpy(Y).to(torch.float32).to(device)
}



lat_fault, lon_fault = 33.0, 134.0
x_fault, y_fault = ll2xy.transform(lon_fault, lat_fault)
print(x_fault, y_fault)

params = {
    "x_fault": torch.tensor(x_fault, requires_grad=True, dtype=torch.float32,  device=device),
    "y_fault": torch.tensor(y_fault, requires_grad=True, dtype=torch.float32, device=device),
    "depth": torch.tensor(15.0, requires_grad=True, dtype=torch.float32, device=device),
    "length": torch.tensor(100, requires_grad=True, dtype=torch.float32, device=device),
    "width": torch.tensor(100, requires_grad=True, dtype=torch.float32, device=device),
    "strike": torch.tensor(270, requires_grad=True, dtype=torch.float32, device=device),
    "dip": torch.tensor(10.0, requires_grad=True, dtype=torch.float32, device=device),
    "rake": torch.tensor(90.0, requires_grad=True, dtype=torch.float32, device=device) ,
    "slip": torch.tensor(5, requires_grad=True, dtype=torch.float32, device=device)
}


import copy
params_init = copy.deepcopy(params)

p1 = {
    "x_fault": torch.tensor(x_fault-100.0),
    "y_fault": torch.tensor(y_fault-100.0),
    "depth": torch.tensor(0.0),
    "length": torch.tensor(20.0),
    "width": torch.tensor(20.0),
    "strike": torch.tensor(0.0),
    "dip": torch.tensor(0.0),
    "rake": torch.tensor(0.0),
    "slip": torch.tensor(0.0)
}

p2 = {
    "x_fault": torch.tensor(x_fault+100.0),
    "y_fault": torch.tensor(y_fault+100.0),
    "depth": torch.tensor(50.0),
    "length": torch.tensor(200.0),
    "width": torch.tensor(200.0),
    "strike": torch.tensor(360.0),
    "dip": torch.tensor(90.0),
    "rake": torch.tensor(360.0),
    "slip": torch.tensor(10.0)
}

def scale(p, p1, p2):
    q = {}
    for key in p:
        q[key] = (
            (p[key] - p1[key]) / (p2[key] - p1[key])
        ).detach().clone().requires_grad_(True)
    return q

def unscale(q, p1, p2):
    p = {}
    for key in q:
        p[key] = p1[key] + (p2[key] - p1[key]) * q[key]
    
    return p


def fault_param_to_corners(params:dict):
    x = params["x_fault"].cpu().detach().numpy() # fault center
    y = params["y_fault"].cpu().detach().numpy()
    length = params["length"].cpu().detach().numpy()
    width = params["width"].cpu().detach().numpy()
    dip = params["dip"].cpu().detach().numpy()
    strike = params["strike"].cpu().detach().numpy()

    ss = np.sin(np.deg2rad(strike))
    cs = np.cos(np.deg2rad(strike))
    sd = np.sin(np.deg2rad(dip))
    cd = np.cos(np.deg2rad(dip))

    x_corners = [
        x - length * ss / 2 - width * cd * cs / 2,
        x - length * ss / 2 + width * cd * cs / 2,
        x + length * ss / 2 + width * cd * cs / 2,
        x + length * ss / 2 - width * cd * cs / 2
    ]
    y_corners = [
        y - length * cs / 2 + width * cd * ss / 2,
        y - length * cs / 2 - width * cd * ss / 2,
        y + length * cs / 2 - width * cd * ss / 2,
        y + length * cs / 2 + width * cd * ss / 2,
    ]

    lon_corners, lat_corners = ll2xy.transform(x_corners, y_corners, direction = "INVERSE")
    corners = np.column_stack((np.array(lon_corners), np.array(lat_corners)))

    return corners



scaled_params = scale(params, p1, p2)
_, _, uz = okada.compute(coords, params, compute_strain=False, is_degree=True, fault_origin="center")
uz_amp = torch.abs(uz).max().item() * 2
print(uz_amp)


optimizer = torch.optim.Adam(
    [p for p in scaled_params.values() if p.requires_grad],
)

def loss_fn(scaled_params):
    params = unscale(scaled_params, p1, p2)

    _, _, uz = okada.compute(coords, params, compute_strain=False, is_degree=True, fault_origin="center")
    x = F.pad(uz.T.reshape(1, Nlon, Nlat), padding) / uz_amp
    out = FNO(x)[2:-3, 2:-3, :] * uz_amp

    loss = (mask * (out - obs) ** 2).sum() / (2 * sigma ** 2)
    
    return loss


def closure():
    loss = loss_fn(scaled_params)
    optimizer.zero_grad()
    loss.backward()
    return loss


epochs = 500

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




best_params =  unscale(scaled_params, p1, p2)
print(best_params)

x_fault = best_params["x_fault"].cpu().detach().numpy()
y_fault = best_params["y_fault"].cpu().detach().numpy()
lon_fault, lat_fault = ll2xy.transform(x_fault, y_fault, direction="INVERSE") 
print(lon_fault, lat_fault)



def plot_array(array, params, fname):
    fig, ax = plt.subplots()
    im = ax.imshow(
        array, vmin=-1, vmax=+1, cmap=cmap, origin="lower",
        extent = [lon_min, lon_max, lat_min, lat_max]
    )

    corners = fault_param_to_corners(params)
    rect = patches.Polygon(corners, ls="--", lw=1.5, closed=True, edgecolor='dimgray', fill=False)
    ax.add_patch(rect)
    ax.plot(    
        [corners[0, 0], corners[3, 0]],
        [corners[0, 1], corners[3, 1]],
        color='dimgray', lw=1.5
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



_, _, uz_init = okada.compute(coords, params_init, compute_strain=False, is_degree=True, fault_origin="center")
x_init = uz_init.to("cpu").detach().numpy()
plot_array(x_init * np.flipud(nanmask), params_init, f"{outdir}/eta0_init.png")

_, _, uz = okada.compute(coords, best_params, compute_strain=False, is_degree=True, fault_origin="center")
x_estimated = uz.to("cpu").detach().numpy()
plot_array(x_estimated * np.flipud(nanmask), best_params, f"{outdir}/eta0.png")

print(np.abs(x_estimated).max())


x_init = F.pad(uz_init.T.reshape(1, Nlon, Nlat), padding) / uz_amp
out_init = FNO(x_init)[2:-3, 2:-3, :] * uz_amp

x = F.pad(uz.T.reshape(1, Nlon, Nlat), padding) / uz_amp
out = FNO(x)[2:-3, 2:-3, :] * uz_amp


ts1 = np.linspace(0, 32, 32+1)
ts2 = np.linspace(1, Nt, Nt)

for station in Stations:
    i, j, true = obs_cache[station]
    pred_init = out_init[i, j, :].to("cpu").detach().numpy()
    pred = out[i, j, :].to("cpu").detach().numpy()

    offset = max(true.max(), pred_init.max(), pred.max()) * 2

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ts1, true, label="obs")
    ax.plot(ts2, pred_init, label="init")
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
    pred_init = out_init[i, j, :].to("cpu").detach().numpy()
    pred = out[i, j, :].to("cpu").detach().numpy()

    if cnt == 0:
        ax.plot(ts1, cnt + true, lw = 2, c = "black", label="Observation")
        ax.plot(ts2, cnt + pred_init, lw = 1, c = "blue", label = "Initial")
        ax.plot(ts2, cnt + pred, lw = 2, c = "red", label = "Inverted")
    else:
        ax.plot(ts1, cnt + true, lw = 2, c = "black")
        ax.plot(ts2, cnt + pred_init, lw = 1, c = "blue")
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




best_params_vec = torch.cat([
    best_params[k].reshape(-1) for k in best_params.keys()
])

def unscaled_loss_fn(params_vec):
    params = {}
    for i, k in enumerate(best_params.keys()):
        params[k] = params_vec[i]

    _, _, uz = okada.compute(coords, params, compute_strain=False, is_degree=True, fault_origin="center")
    x = F.pad(uz.T.reshape(1, Nlon, Nlat), padding) / uz_amp
    out = FNO(x)[2:-3, 2:-3, :] * uz_amp

    loss = (mask * (out - obs) ** 2).sum() / (2 * sigma ** 2)
    
    return loss


def compute_hessian(loss_fn, params):
    n = len(params)
    H = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    loss = loss_fn(params)
    grads = torch.autograd.grad(loss, params, create_graph=True)[0]
    
    for i in range(n):
        H[i] = torch.autograd.grad(grads[i], params, retain_graph=True)[0]
    
    return H

H = compute_hessian(unscaled_loss_fn, best_params_vec)
print(H)


Sigma = torch.linalg.inv(H)
print(Sigma)

Variance = torch.diagonal(Sigma)
print(Variance)

Std = torch.sqrt(Variance)
print(Std)



x_std, y_std = Std[0].item(), Std[1].item()

lon_std = x_std / (111 * np.cos(lat_fault))
lat_std = y_std / 111

print(lon_std, lat_std)
