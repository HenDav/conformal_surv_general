from torch import nn
import torch
from itertools import chain

class MySoftPlus(torch.nn.Module):
    def __init__(self, low):
        super().__init__()
        self.low = low
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x):
        log_h = self.softplus(x) + self.low
        return log_h

def create_mlp(d_in, d_hids, d_out, norm, p, activation='relu', dtype=torch.float32):
    act_fn = {
        'relu':nn.ReLU(),
        'elu':nn.ELU(),
        'selu':nn.SELU(),
        'silu':nn.SiLU()
        }

    act_fn = act_fn[activation]

    norm_fn = {
        'layer':nn.LayerNorm,
        'batch':nn.BatchNorm1d
        }

    if norm in norm_fn.keys():
        norm_fn = norm_fn[norm]
    else:
        norm = False

    net = list(
            chain(
                *[
                    [
                        nn.Linear(
                            d_in if ii == 0 else d_hids[ii],
                            d_out if ii + 1 == len(d_hids) else d_hids[ii],
                            dtype=dtype
                        ),
                        nn.Identity() if ii + 1 == len(d_hids) else act_fn,
                        nn.Identity() if not norm else norm_fn(d_hids[ii], dtype=dtype),
                        nn.Dropout(p) if ii + 1 != len(d_hids) else nn.Identity()
                    ]
                    for ii in range(len(d_hids))
                ]
            )
        )
    net.pop(-1)
    net.pop(-1)
    net = nn.Sequential(*net)

    return net