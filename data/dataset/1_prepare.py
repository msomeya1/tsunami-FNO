import os
import numpy as np
import pyproj
import xarray as xr

def my_round(x):
    return int((2*x+1)//2)


lon_min, lon_max = 131.2, 136.2
lat_min, lat_max = 29.5, 34.5
Nlon, Nlat = 251, 251
dlon, dlat = 0.02, 0.02

lon_mid, lat_mid = (lon_min+lon_max)/2, (lat_min+lat_max)/2
ll2xy = pyproj.Transformer.from_crs(
    crs_from="EPSG:4326", # WGS84
    crs_to=f"+proj=tmerc +lon_0={lon_mid} +lat_0={lat_mid} +ellps=WGS84 +datum=WGS84 +units=km", 
    always_xy=True
)

grd = xr.open_dataset("../../share/Hyuganada.grd")
bathy = grd.z.data.reshape(Nlat,Nlon)
bathy = np.rot90(bathy, k=-1)

def generate_gridfile_dat():
    with open("gridfile.dat", mode="w") as f:
        f.write(f'SD01 SD01 1 "../../../share/Hyuganada.grd" NO_DISPLACEMENT_FILE_GIVEN')


def generate_tsun_par_fault_params():
    with open("tsun.par", mode="w") as f:
        f.write('&params\n')
        f.write('gridfile="gridfile.dat"\n')
        f.write('tgstafn="../../../share/Stations_DONET2Nnet.jagurs"\n')
        f.write('dt=1\n')
        f.write('tend=1920\n')
        f.write('itmap=60\n')
        f.write('init_disp_fault=1\n')
        f.write('fault_param_file="fault_param.txt"\n')
        f.write('tau=0\n')
        f.write('cf=0\n')
        f.write('cfl=0\n')
        f.write('coriolis=0\n')
        f.write('velgrd=0\n')
        f.write('/')

def generate_fault_paramter_file(i_sample):

    def check_inside_domain(lat_center, lon_center, depth_center, length, width, strike, dip):
        dx_strike = length * np.sin(np.deg2rad(strike))
        dy_strike = length * np.cos(np.deg2rad(strike))
        dx_dip =  width * np.cos(np.deg2rad(dip)) * np.cos(np.deg2rad(strike))
        dy_dip = -width * np.cos(np.deg2rad(dip)) * np.sin(np.deg2rad(strike))
        dz_dip =  width * np.sin(np.deg2rad(dip))

        x_center, y_center = ll2xy.transform(lon_center, lat_center)

        # top left
        x_topleft = x_center - 0.5 * (dx_strike + dx_dip)
        y_topleft = y_center - 0.5 * (dy_strike + dy_dip)
        dep_topleft = depth_center - 0.5 * dz_dip

        # top right
        x_topright = x_center + 0.5 * (dx_strike - dx_dip)
        y_topright = y_center + 0.5 * (dy_strike - dy_dip)

        # bottom right
        x_botright = x_center + 0.5 * (dx_strike + dx_dip)
        y_botright = y_center + 0.5 * (dy_strike + dy_dip)

        # bottom left
        x_botleft = x_center - 0.5 * (dx_strike - dx_dip)
        y_botleft = y_center - 0.5 * (dy_strike - dy_dip)

        x_nodes = [x_center, x_topleft, x_topright, x_botright, x_botleft]
        y_nodes = [y_center, y_topleft, y_topright, y_botright, y_botleft]
        lon_nodes, lat_nodes = ll2xy.transform(x_nodes, y_nodes, direction="INVERSE")
        lon_center, lon_topleft, lon_topright, lon_botright, lon_botleft = lon_nodes
        lat_center, lat_topleft, lat_topright, lat_botright, lat_botleft = lat_nodes

        inside_domain = True

        for lon, lat in zip(lon_nodes, lat_nodes):
            if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                i = my_round((lon-lon_min)/dlon)
                j = my_round((lat-lat_min)/dlat)
                if bathy[i, j] < 0:
                    inside_domain *= False
            else:
                inside_domain *= False

        inside_domain *= (dep_topleft > 5)

        return inside_domain, lat_topleft, lon_topleft, dep_topleft

    with open("fault_param.txt", 'w') as f:
        f.write("! Rectangular fault model generated from random fault parameters\n")
        f.write("! tau=0\n")
        f.write("! lat[deg], lon[deg], depth[km], length[km], width[km], dip[deg], strike[deg], rake[deg], slip[m]\n")

        # Sample the center coordinate so that it falls in the domain and in the ocean (depth > 0)
        # Calculate the coordinates of the four vertices based on length, width, etc.
        # Check that all of them are also within the ocean area and do not protrude above the surface

        rng = np.random.default_rng(i_sample)

        for _ in range(100):

            lat_center = rng.uniform(lat_min, lat_max)
            lon_center = rng.uniform(lon_min, lon_max)
            depth_center = rng.uniform(0, 50)
            length = rng.uniform(20, 200)
            width = length / 2
            dip = rng.uniform(0, 90)
            strike = rng.uniform(0, 360)
            rake = rng.uniform(0, 360)
            slip = rng.uniform(0, 10)

            inside_domain, lat, lon, depth = check_inside_domain(lat_center, lon_center, depth_center, length, width, strike, dip)

            if inside_domain:
                f.write(f"{lat:.4f} {lon:.4f} {depth:.4f} {length:.4f} {width:.4f} {dip:.4f} {strike:.4f} {rake:.4f} {slip:.4f}\n")
                break

        else:
            print(f"cannot find paramters (sample {i_sample})")

Ntrain = 2000
Nvalid = 200

sample = 1

for i in range(Ntrain+Nvalid):
    sim_dir = f"sample{sample}"
    os.makedirs(sim_dir, exist_ok=True)
    os.chdir(sim_dir)

    generate_gridfile_dat()
    generate_tsun_par_fault_params()
    generate_fault_paramter_file(i)
    
    os.chdir("..")
    sample += 1
