import pandas as pd
import numpy as np
import xarray as xr
import pyproj
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
import matplotlib.ticker as ticker
import matplotlib.patches as patches

lon_min, lon_max = 131.2, 136.2
lat_min, lat_max = 29.5, 34.5
Nlon, Nlat = 251, 251

lon_mid, lat_mid = (lon_min+lon_max)/2, (lat_min+lat_max)/2

ll2xy = pyproj.Transformer.from_crs(
    crs_from="EPSG:4326", # WGS84
    crs_to=f"+proj=tmerc +lon_0={lon_mid} +lat_0={lat_mid} +ellps=WGS84 +datum=WGS84 +units=km", 
    always_xy=True
)


grd = xr.open_dataset("../../share/Hyuganada.grd")
bathy = grd.z.data.reshape(Nlat,Nlon)

nanmask = np.ones_like(bathy, dtype=float)
nanmask[bathy <= 0] = np.nan

import matplotlib as mpl
cmap = mpl.colormaps.get_cmap("seismic")
cmap.set_bad(color='gray')


scale = 1

for i in [47, 69, 399]:
    sim_dir = f"sample{i}"
    grd = xr.open_dataset(f"{sim_dir}/SD01.nc")

    trange = [0, 32]


    df = pd.read_csv(
        f"{sim_dir}/fault_param.txt", comment="!", sep="\s+",
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

    


    for it in trange:
        x = grd.wave_height.data[it,:,:]


        fig, ax = plt.subplots()
        im = ax.imshow(
            x, vmin=-scale, vmax=+scale, cmap=cmap, origin="lower",
            extent = [lon_min, lon_max, lat_min, lat_max]
        )

        rect = patches.Polygon(corners, ls="--", lw=1.5, closed=True, edgecolor='dimgray', fill=False)
        ax.add_patch(rect)
        ax.plot(    
            [corners[0, 0], corners[3, 0]],
            [corners[0, 1], corners[3, 1]],
            color='dimgray', lw=1.5
        )


        for spine in ax.spines.values():
            spine.set_linewidth(2)

        
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(which="both", top=True, right=True)
        
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d°E'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°N'))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height [m]")
        plt.savefig(f"sample{i}-{it}.png", dpi=400)