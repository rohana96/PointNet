import torch
import torch.nn as nn


# ------ TO DO ------
class PointNet(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, points):
        """
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        """
        batch_size, num_pts, _ = points.shape

        local_feat = self.relu(self.fc1(points))
        out = self.relu(self.fc2(local_feat))
        out = self.relu(self.fc3(out))

        out = out.view(batch_size, num_pts, -1)
        global_feat, _ = torch.max(out, dim=1)

        out = self.relu(self.fc4(global_feat))
        out = self.relu(self.fc5(out))
        out = self.fc6(out)
        cls_scores = out #self.softmax(out)
        return local_feat, global_feat, cls_scores


# ------ TO DO ------
class PointNetSeg(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(PointNetSeg, self).__init__()

        self.pointnet_cls = PointNet() 
        self.fc1 = nn.Linear(1088, 512)      
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_seg_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, points):
        """
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        """

        batch_size, num_pts, _ = points.shape
        local_feat, global_feat, cls_scores = self.pointnet_cls(points)
        seg_feat = torch.cat([local_feat, global_feat[:, None, :].repeat(1, num_pts, 1)], dim=-1)
        out = self.relu(self.fc1(seg_feat))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        seg_scores = out #self.softmax(out)
        return local_feat, global_feat, seg_scores


def test_pointnet():
    x = torch.rand(size=(2, 10, 3))
    model = PointNet()
    print(model(x))

def test_pointnet_seg():
    x = torch.rand(size=(2, 10, 3))
    model = PointNetSeg()
    print(model(x))

if __name__ == "__main__":
    test_pointnet()
    test_pointnet_seg()

