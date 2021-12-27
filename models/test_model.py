import torch.nn as nn 

def conv_block(in_f, out_f, k=3, s=2, padding='valid'):
    return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=k, stride=s, padding=padding),
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
        )
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 1)
        self.fc = nn.Linear(6241, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        output = self.conv1(X)
        output = self.conv2(output)
        output = nn.Flatten(start_dim=-3, end_dim=-1)(output)
        output = self.fc(output)
        return self.sigmoid(output)