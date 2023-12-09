import os
import torch

package = torch.load("./38_dice_0.76114_ema.pt")
p = './38_dice_0.76114_ema.pth'
torch.save(package.state_dict(), p)