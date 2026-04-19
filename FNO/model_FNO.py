import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce



class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (2 * in_channels))**(1.0 / 2.0)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x, y, t) * (in_channel, out_channel, x, y, t) -> (batch, out_channel, x, y, t)
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights)


    def forward(self, x):
        batchsize, _, d1, d2, d3 = x.shape

        #Compute Fourier coeffcients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, d1, d2, d3//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1 , :self.modes2 , :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1 , :self.modes2 , :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2 , :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2 , :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1 , -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1 , -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(d1, d2, d3), norm = 'forward')
        return x
    




class TsunamiFNO(nn.Module):
    def __init__(self, Nt, width, modes1, modes2, modes3, bathy_data):
        super(TsunamiFNO, self).__init__()

        self.Nt = Nt

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.fc0 = nn.Linear(5, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)

        self.register_buffer('bathy', bathy_data.detach().clone().float(), persistent=False) # (Nx, Ny)


    def forward(self, x):
        batch, Nx, Ny = x.shape
        
        # (b, Nx, Ny) -> (b, Nx, Ny, Nt, 1)
        x = x.reshape(batch, Nx, Ny, 1, 1).expand(batch, Nx, Ny, self.Nt, 1)
        # (Nx, Ny) -> (b, Nx, Ny, Nt, 1)
        bathy = self.bathy.reshape(1, Nx, Ny, 1, 1).expand(batch, Nx, Ny, self.Nt, 1)
        grid = self.get_grid(batch, Nx, Ny, self.Nt, x.device) # (b, Nx, Ny, Nt, 3)

        x = torch.cat((x, bathy, grid), dim=-1) # (b, Nx, Ny, Nt, 5)

        x = self.fc0(x)

        # (b, Nx, Ny, Nt, w) -> (b, w, Nx, Ny, Nt)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batch, self.width, -1)).view(batch, self.width, Nx, Ny, self.Nt)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batch, self.width, -1)).view(batch, self.width, Nx, Ny, self.Nt)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batch, self.width, -1)).view(batch, self.width, Nx, Ny, self.Nt)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batch, self.width, -1)).view(batch, self.width, Nx, Ny, self.Nt)
        x = x1 + x2

        # (b, w, Nx, Ny, Nt) -> (b, Nx, Ny, Nt, w)
        x = x.permute(0, 2, 3, 4, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # (b, Nx, Ny, Nt, 1) -> (b, Nx, Ny, Nt)
        x = x.squeeze()

        return x


    def get_grid(self, batch, Nx, Ny, Nt, device):
        gridx = torch.linspace(0, 1, Nx, device=device)
        gridy = torch.linspace(0, 1, Ny, device=device)
        gridz = torch.linspace(0, 1, Nt, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(gridx, gridy, gridz, indexing='ij')
        
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1, -1)





def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c