
import torch.nn as nn

class RefinedCRAFT(nn.Module):
    def __init__(self, base_net, refine_net):
        super(RefinedCRAFT, self).__init__()
        self.base_net = base_net
        self.refine_net = refine_net

    def forward(self, x):
        base_matrix, feature = self.base_net(x)
        refined_matrix = self.refine_net(base_matrix, feature)
        modified_matrix = base_matrix[:, :, :, :1].squeeze(dim=3)
        refined_matrix = refined_matrix.squeeze(dim=3)
        return modified_matrix, refined_matrix
