import torch
import torch.nn as nn
import numpy as np
import math

def compl_mul1d(order, x, weights):
    return torch.einsum('bhi,hio->bho', x, weights)

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()

        """
        fourier layer
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.mode_select_method = mode_select_method
        # Can be changed to other types
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        
        # Compute Fourier coeffcients up to factor of modes
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, H, E, L//2 + 1, dtype=torch.cfloat, device=x.device)
        
        if self.mode_select_method == 'random':
            # Random mode selection
            modes = min(self.modes1, L//2+1)
            if modes > 0:
                index_list = list(range(0, L//2+1))
                np.random.shuffle(index_list)
                index_list = index_list[:modes]
            else:
                index_list = []
        elif self.mode_select_method == 'low':
            # Low frequency modes
            modes = min(self.modes1, L//2+1)
            index_list = list(range(0, modes))
        else:
            raise ValueError('mode_select_method should be random or low')
        
        for i in range(min(self.modes1, len(index_list))):
            mode_idx = index_list[i]
            out_ft[:, :, :, mode_idx] = self.compl_mul1d(x_ft[:, :, :, mode_idx], self.weights1[i])
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=L, dim=-1)
        x = x.permute(0, 3, 1, 2)  # size = [B, L, H, E]
        return (x, None)

class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print('fourier enhanced cross attention used!')
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.mode_select_method = mode_select_method
        # Can be changed to other types
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1) # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coeffcients up to factor of modes
        xq_ft_ = torch.zeros(B, H, E, L//2 + 1, dtype=torch.cfloat, device=xq.device)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        
        # Randomly select modes or low frequency modes
        if self.mode_select_method == 'random':
            modes = min(self.modes1, L//2+1)
            if modes > 0:
                index_list = list(range(0, L//2+1))
                np.random.shuffle(index_list)
                index_list = index_list[:modes]
            else:
                index_list = []
        elif self.mode_select_method == 'low':
            modes = min(self.modes1, L//2+1)
            index_list = list(range(0, modes))
        else:
            raise ValueError('mode_select_method should be random or low')
        
        for i in range(min(self.modes1, len(index_list))):
            mode_idx = index_list[i]
            xq_ft_[:, :, :, mode_idx] = self.compl_mul1d(xq_ft[:, :, :, mode_idx], self.weights1[i])

        xqo_ft = xq_ft_
        xqo = torch.fft.irfft(xqo_ft, n=L, dim=-1)
        xqo = xqo.permute(0, 3, 1, 2)  # size = [B, L, H, E]

        return (xqo, None) 