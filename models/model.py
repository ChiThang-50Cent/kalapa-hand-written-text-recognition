import torch
import torch.nn as nn
import torchvision.models as models

class VGG16_FeatureExtractor(nn.Module):
    def __init__(self, in_channel):
        super(VGG16_FeatureExtractor, self).__init__()

        vgg16 = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                               out_channels=64, kernel_size=3, 
                               padding=1, stride=1),
            *list(vgg16.children())[0][1:23]
            )
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 1), stride=(3, 1), padding=(0, 0))
        # for layer in self.vgg:
        #     if not isinstance(layer, nn.MaxPool2d):
        #         for param in layer.parameters():
        #             param.requires_grad_(False)
    
    def forward(self, X: torch.Tensor):
        X = X.type(torch.float)
        out = self.vgg(X)
        out = self.maxpool(out)

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

        b, c, h, w = self.extractor(torch.randn(1, in_channels, imgSize[0], imgSize[1])).shape

        self.eca = ECALayer()
        self.sequential = nn.Sequential(
            BidirectionalLSTM(c * h, hidden_dim, hidden_dim),
            BidirectionalLSTM(hidden_dim, hidden_dim, numChars),
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

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        
        self.eca = ECALayer()

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=True):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 256x4x16
        convRelu(4)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 0)))  # 512x2x16
        convRelu(6)  # 512x1x16
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((3, 1), (3, 1), (0, 0)))
        
        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, 0),
            BidirectionalLSTM(nh, nh, nclass, 0)
            )

    def forward(self, input):
        # conv features
        input = input.type(torch.float)
        conv = self.cnn(input)
        eca = self.eca(conv)
        conv = conv + eca

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = torch.nn.functional.log_softmax(output, 2)

        return output


if __name__ == '__main__':

    model = Model((64, 768), 'vgg16', 1, 256, 200)
    # model = CRNN(imgH=64, nc=1, nclass=199, nh=256)
    out = model(torch.randn(1, 1, 64, 768))

    print(out.shape)
    # print(model)

