import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice_coeff = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        dice_loss = 1 - dice_coeff

        return dice_loss