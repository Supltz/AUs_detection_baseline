import torch
from ResNets import ResNet34
import torch.nn as nn


class L_Net(ResNet34):
    def __init__(self, num_classes):
        super(L_Net,self).__init__(num_classes)
        self.lstm_list = []
        for i in range(num_classes):
            self.lstm = nn.LSTM(512, 49, 1, dropout=0.2).to('cuda:1')
            self.lstm_list.append(self.lstm)

    def init(self,x):
        h0 = torch.empty(1, x.shape[0], 49).to('cuda:1')  #初始化参数（非必要）
        c0 = torch.empty(1, x.shape[0], 49).to('cuda:1')
        nn.init.xavier_normal_(h0,gain=20)
        nn.init.xavier_normal_(c0,gain=20)

    def reshape_data(self,x):
        x = x.reshape([x.shape[0], 512, 49])  # 128x512x49
        x = x.transpose(0, 2)  # 49x512x128
        input = x.transpose(1, 2)  # 49x128x512
        return input

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        self.init(x)
        input=self.reshape_data(x)
        out=torch.zeros([input.shape[1],12]).to('cuda:1')   #LSTM
        for i in range(12):
            output,(hn,cn)=self.lstm_list[i](input)
            hn=hn.mean(2)#128x1
            out[:,i]=torch.add(out[:,i],hn)
        return out  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12