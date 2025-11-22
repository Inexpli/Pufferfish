import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(8 * 8 * 256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class SmallResNetValue(nn.Module):
    def __init__(self, in_channels=18, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(base_channels*4, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)  
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x
