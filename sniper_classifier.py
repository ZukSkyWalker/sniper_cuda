import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
  def __init__(self, k):
    super().__init__()
    self.k = k
    self.conv1 = nn.Conv1d(k, 64, 1)
    self.conv2 = nn.Conv1d(64, 256, 1)
    self.fc1 = nn.Linear(256, 128)
    self.fc2 = nn.Linear(128, k*k)

    self.bn1 = nn.BatchNorm1d(64, momentum=0.1)
    self.bn2 = nn.BatchNorm1d(256, momentum=0.1)
    self.bn3 = nn.BatchNorm1d(128, momentum=0.1)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(batch_size, -1)
    x = F.relu(self.bn3(self.fc1(x)))
    x = self.fc2(x)
    x += torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
    return x.view(batch_size, self.k, self.k)

class SniperNet(nn.Module):
  def __init__(self, num_classes=4):
    super(SniperNet, self).__init__()
    self.num_classes = num_classes
    self.input_transform = TNet(k=3)
    self.feature_transform = TNet(k=32)
    self.conv1 = nn.Conv1d(3, 32, 1)
    self.conv2 = nn.Conv1d(32, 32, 1)
    self.conv3 = nn.Conv1d(32, 64, 1)

    self.fc1 = nn.Linear(64, 128)
    self.fc2 = nn.Linear(128, num_classes)

    self.bn1 = nn.BatchNorm1d(32, momentum=0.1)
    self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
    self.bn3 = nn.BatchNorm1d(128, momentum=0.1)


  def forward(self, x):
    batch_size = x.size(0)
    x = x.transpose(1, 2)
    input_transform = self.input_transform(x)

    # batch matrix-matrix product: (b, n, m) * (b, m, p) => (b, n, p)
    x = torch.bmm(x.transpose(1, 2), input_transform).transpose(1, 2)

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.conv2(x))

    feature_transform = self.feature_transform(x)
    x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)
    x = F.relu(self.bn2(self.conv3(x)))
    # x = nn.MaxPool1d(x.size(-1))(x)
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(batch_size, -1)
    x = F.relu(self.bn3(self.fc1(x)))
    x = self.fc2(x)

    return x, input_transform, feature_transform
