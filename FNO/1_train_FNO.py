import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from timeit import default_timer
from model_FNO import TsunamiFNO, count_params


class config:
    indir = "../../data/dataset"
    outdir = "out"
    bathy_file = "../../share/Hyuganada.grd"
    Ntrain = 2000
    Nvalid = 200
    Nlon = 251
    Nlat = 251
    Nt = 32
    modes_x = 64 # less than Nlon
    modes_y = 64 # less than Nlat
    modes_t = 16 # less than Nt//2+1
    width = 16
    batch_size = 4
    epochs = 500
    seed = 123



class LpLoss(object):
    def __init__(self, p=2):
        super(LpLoss, self).__init__()
        self.p = p

    def __call__(self, x, y):
        batch_size = x.shape[0]

        diff_norms = torch.norm(x.reshape(batch_size, -1) - y.reshape(batch_size, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(batch_size, -1), self.p, 1)

        return torch.sum(diff_norms / y_norms)
    


def load_data():
        
    t1 = default_timer()

    Nx = config.Nlon + 5
    Ny = config.Nlat + 5
    Nevent = config.Ntrain + config.Nvalid
    data_all = np.zeros((Nevent, Nx, Ny, config.Nt+1))

    # read netCDF files
    for ievent in range(Nevent):
        grd = xr.open_dataset(f"{config.indir}/sample{ievent+1}/SD01.nc")
        data = np.nan_to_num(grd.wave_height.data[:config.Nt+1, :, :]) 
        amp = np.abs(data[0, :, :]).max() * 2
        # [Nt+1, Nlat, Nlon] -> [Nlon, Nlat, Nt+1]
        data_all[ievent, 2:-3, 2:-3, :] = data.transpose(2, 1, 0) / amp 

        if ievent % 50 == 0:
            print(ievent, flush=True)

    # transform ndarray into torch tensor
    data_all = torch.from_numpy( data_all.astype(np.float32) )

    # divide train/valid
    train_a = data_all[:config.Ntrain, :, :, 0]
    train_u = data_all[:config.Ntrain, :, :, 1:]
    valid_a = data_all[-config.Nvalid:, :, :, 0]
    valid_u = data_all[-config.Nvalid:, :, :, 1:]

    print("shape of tensor:")
    print(train_a.shape)
    print(valid_a.shape)
    print(train_u.shape)
    print(valid_u.shape)


    train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
    valid_dataset = torch.utils.data.TensorDataset(valid_a, valid_u)

    t2 = default_timer()
    print("Time required for data-loading = ", t2-t1, "[s]")

    return train_dataset, valid_dataset



def train(device, FNO, optimizer, scheduler, train_loader, Ntrain):

    L1_fn = LpLoss(p=1)
    L2_fn = LpLoss(p=2)

    L = 0.0
    L1 = 0.0
    L2 = 0.0

    FNO.train()
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = FNO(x)

        l1 = L1_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
        l2 = L2_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
        loss = 0.1 * l1 + 0.9 * l2

        loss.backward()

        L += loss.item()
        L1 += l1.item()
        L2 += l2.item()

        optimizer.step()

    scheduler.step()


    L /= Ntrain
    L1 /= Ntrain
    L2 /= Ntrain

    return L, L1, L2



def valid(device, FNO, valid_loader, Nvalid):

    L1_fn = LpLoss(p=1)
    L2_fn = LpLoss(p=2)

    L = 0.0
    L1 = 0.0
    L2 = 0.0

    FNO.eval()
    
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)

        out = FNO(x)

        l1 = L1_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
        l2 = L2_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
        loss = 0.1 * l1 + 0.9 * l2

        L += loss.item()
        L1 += l1.item()
        L2 += l2.item()
            
    L /= Nvalid
    L1 /= Nvalid
    L2 /= Nvalid

    return L, L1, L2



def main(device, train_dataset, valid_dataset):

    t1 = default_timer()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False
    )


    ## bathymetry
    Nx = config.Nlon + 5
    Ny = config.Nlat + 5
    bathy = np.zeros((Nx, Ny), dtype=np.float32)
    grd = xr.open_dataset(config.bathy_file)
    bathy0 = np.rot90( grd.z.data.reshape(config.Nlat, config.Nlon), k=-1) # -> (Nlon, Nlat)
    bathy0[bathy0<0.0] = 0.0
    bathy[2:-3, 2:-3] = bathy0
    bathy /= bathy.max()
    bathy = torch.tensor(bathy)

    
    
    # build FNO model
    FNO = TsunamiFNO(config.Nt, config.width, config.modes_x, config.modes_y, config.modes_t, bathy).to(device)
    
    optimizer = torch.optim.Adam(FNO.parameters(), lr=1e-3, weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    
    os.makedirs(f"{config.outdir}/model", exist_ok=True)
    f = open(f'{config.outdir}/loss_history.txt', 'w')
    print("# of params = ", count_params(FNO))
    print("epoch, time, L_train, L1_train, L2_train, L_valid, L1_valid, L2_valid")
    print("# epoch, time, L_train, L1_train, L2_train, L_valid, L1_valid, L2_valid", flush=True, file=f)


    for epoch in range(config.epochs):

        t2 = default_timer()

        L_train, L1_train, L2_train = train(
            device, FNO, optimizer, scheduler, train_loader, config.Ntrain
        )
        L_valid, L1_valid, L2_valid = valid(
            device, FNO, valid_loader, config.Nvalid
        )

        t3 = default_timer()

        print(epoch, t3-t2, L_train, L1_train, L2_train, L_valid, L1_valid, L2_valid, flush=True)
        print(epoch, t3-t2, L_train, L1_train, L2_train, L_valid, L1_valid, L2_valid, flush=True, file=f)


        if epoch % 10 == 0:
            torch.save(FNO.state_dict(), f'{config.outdir}/model/model_{epoch}.pt')


    t4 = default_timer()


    print("\n")
    print("\n", file=f)
    print("Time required for training = ", t4-t1, "[s]")
    f.close()


    torch.save(FNO.state_dict(), f'{config.outdir}/model.pt')
    
    


    #--------------- plot loss history ---------------#
    loss_history = np.loadtxt(f'{config.outdir}/loss_history.txt')
    steps = loss_history[:, 0]
    L_train = loss_history[:, 2] 
    L_valid = loss_history[:, 5] 

    fig, ax = plt.subplots(figsize=(8,5), dpi=200)
    ax.semilogy(steps, L_train, label="train")
    ax.semilogy(steps, L_valid, label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    ax.legend()
    plt.savefig(f"{config.outdir}/loss_history.png")
    plt.close()






    
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(f"output directory: {config.outdir}")
    os.makedirs(config.outdir, exist_ok=True)

    train_dataset, valid_dataset = load_data()
    main(device, train_dataset, valid_dataset)

    print("done.")