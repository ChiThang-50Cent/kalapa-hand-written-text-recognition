import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision.models as models

class VGG16_FeatureExtractor(nn.Module):
    def __init__(self, in_channel):
        super(VGG16_FeatureExtractor, self).__init__()

        vgg16 = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                               out_channels=64, kernel_size=3, 
                               padding=1, stride=1),
            *list(vgg16.children())[0][1:17]
            )
    
    def forward(self, X):
        out = self.vgg(X)

        return out

class ECALayer(nn.Module):

    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        recurrent = self.dropout(recurrent)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class Model(nn.Module):
    def __init__(self, imgSize, backbone, in_channels, hidden_dim, numChars):
        super(Model, self).__init__()
        
        if backbone == 'vgg16':
            self.extractor = VGG16_FeatureExtractor(in_channel=in_channels)
        with torch.no_grad():
            b, c, h, w = self.extractor(torch.randn(1, in_channels, imgSize[0], imgSize[1])).shape

        self.eca = ECALayer()
        self.sequential = nn.Sequential(
            nn.Linear(c * h , hidden_dim),
            BidirectionalLSTM(hidden_dim, hidden_dim, hidden_dim),
            BidirectionalLSTM(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, numChars)
        )
    
    def forward(self, x):
        output = self.extractor(x)
        eca = self.eca(output)

        output = output + eca

        B, C, H, T = eca.shape

        output = output.reshape(B, T, C * H)
        output = self.sequential(output)

        
        output = output.permute(1, 0, 2)

        output = torch.nn.functional.log_softmax(output, 2)

        return output
    
if __name__ == '__main__':

    model = Model((64, 768), 'vgg16', 1, 256, 200)
    out = model(torch.randn(1, 1, 64, 768))

    print(out.shape)
    # print(model)

