import torch
import torch.nn as nn


# ------ TO DO ------
class PointNet(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNet, self).__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList([
            nn.Linear(3, 64),
            nn.Linear(64, 128),
            nn.Linear(64, 1024),
            nn.MaxPool1d(kernel_size=1),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        ])

    def forward(self, points):
        """
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        """
        batch_size, num_pts, _ = points.shape
        out = points
        out = out.reshape(-1, 3)
        for layer in self.layers:
            # import pdb
            # pdb.set_trace()
            out = layer(out)
        return out.reshape(batch_size, self.num_classes)
        pass

def test_pointnet():
    x = torch.rand(size=(2, 10, 3))
    model = PointNet()
    print(model(x).shape)


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()
        pass

    def forward(self, points):
        """
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        """
        pass


if __name__ == "__main__":
    test_pointnet()
    print("hello")
