
import torch
from torch import nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
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
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)



        return output

def create_model(config):
    crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['num_of_outputs'], 512)
    return crnn

# import torch
# from torch import nn
# import torch.nn.functional as F
# class BidirectionalLSTM(nn.Module):

#     def __init__(self, nIn, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()

#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
#         self.embedding = nn.Linear(nHidden * 2, nOut)

#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         # print(input.shape,"\n", recurrent.shape)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)

#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#         # print(output.shape)

#         return output

# class CRNN(nn.Module):

#     def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False):
#         super(CRNN, self).__init__()
#         # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

#         ks = [3, 3, 3, 3, 3, 3, 2]
#         ps = [1, 1, 1, 1, 1, 1, 0]
#         ss = [1, 1, 1, 1, 1, 1, 1]
#         nm = [64, 128, 256, 256, 512, 512, 512]

#         cnn = nn.Sequential()

#         def convRelu(i, batchNormalization=False):
#             nIn = nc if i == 0 else nm[i - 1]
#             nOut = nm[i]
#             cnn.add_module('conv{0}'.format(i),
#                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
#             if batchNormalization:
#                 cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut,track_running_stats=True))
#             if leakyRelu:
#                 cnn.add_module('relu{0}'.format(i),
#                                nn.LeakyReLU(0.2, inplace=True))
#             else:
#                 cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

#         convRelu(0)
#         cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
#         convRelu(1)
#         cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
#         convRelu(2, True)
#         convRelu(3)
#         cnn.add_module('pooling{0}'.format(2),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
#         convRelu(4, True)
#         convRelu(5)
#         cnn.add_module('pooling{0}'.format(3),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
#         convRelu(6, True)  # 512x1x16

#         self.cnn = cnn
#         # self.rnn = nn.Sequential(
#         #     BidirectionalLSTM(512, nh, nh),
#         #     BidirectionalLSTM(nh, nh, nclass))

#         self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
#         self.softmax = nn.LogSoftmax(2)

#     # def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False):
#     #     super(CRNN, self).__init__()
#     #     # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

#     #     ks = [3, 3, 3, 3, 3, 3, 2]
#     #     ps = [1, 1, 1, 1, 1, 1, 0]
#     #     ss = [1, 1, 1, 1, 1, 1, 1]
#     #     nm = [64, 128, 256, 256, 512, 512, 512]

#     #     cnn = nn.Sequential()

#     #     def convRelu(i, batchNormalization=False):
#     #         nIn = nc if i == 0 else nm[i - 1]
#     #         nOut = nm[i]
#     #         cnn.add_module('conv{0}'.format(i),
#     #                        nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
#     #         if batchNormalization:
#     #             cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut,track_running_stats=True))
#     #         if leakyRelu:
#     #             cnn.add_module('relu{0}'.format(i),
#     #                            nn.LeakyReLU(0.2, inplace=True))
#     #         else:
#     #             cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

#     #     convRelu(0)
#     #     cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
#     #     convRelu(1)
#     #     cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
#     #     convRelu(2, True)
#     #     convRelu(3)
#     #     cnn.add_module('pooling{0}'.format(2),
#     #                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
#     #     convRelu(4, True)
#     #     convRelu(5)
#     #     cnn.add_module('pooling{0}'.format(3),
#     #                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
#     #     convRelu(6, True)  # 512x1x16

#     #     self.cnn = cnn
#     #     # self.rnn = nn.Sequential(
#     #     #     BidirectionalLSTM(512, nh, nh),
#     #     #     BidirectionalLSTM(nh, nh, nclass))

#     #     self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
#     #     self.softmax = nn.LogSoftmax(2)


#     def forward(self, input):
#         # conv features
#         conv = self.cnn(input)
#         # print("a:", conv.shape)
#         b, c, h, w = conv.size()
#         assert h == 1, "the height of conv must be 1"
#         # conv = conv.squeeze(2)
#         conv = conv.view(b, -1, w)
#         # print("b:", conv.shape)
#         conv = conv.permute(2, 0, 1)  # [w, b, c]
#         # print("c:", conv.shape)
#         # rnn features
#         output = self.rnn(conv)
#         # print("d:", output.shape)
#         # output = F.log_softmax(output, dim=2)
#         output = F.log_softmax(output, dim=2)

#         return output
#     def backward_hook(self, module, grad_input, grad_output):
#         for g in grad_input:
#             g[g != g] = 0   # replace all nan/inf in gradients to zero

# def create_model(config):
#     crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['num_of_outputs'], 512)
#     crnn.register_backward_hook(crnn.backward_hook)

#     return crnn
