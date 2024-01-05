import os
import torch




class LocationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        
        KFL_gt = torch.nonzero(target).squeeze()
        
        
        KFL_pred = torch.sigmoid(prediction)
        
       
        squared_diff = torch.pow(target - KFL_pred, 2)

        
        temp =  torch.arange(len(KFL_pred)).to('cuda')
        diff = torch.pow(KFL_gt - temp, 2)
        
        squared_diff = torch.exp(-diff) * squared_diff
        
    
        


        return sum(squared_diff)




class KeyFrameLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ClsLoss = torch.nn.BCEWithLogitsLoss()
        self.LocationLoss = LocationLoss()

    def forward(self, prediction, target):
        loss = self.LocationLoss(prediction, target)
        # loss = self.ClsLoss(prediction, target) + self.LocationLoss(prediction, target)
        return loss

