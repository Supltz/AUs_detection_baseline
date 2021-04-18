import torch
import torch.nn as nn
from ResNets import ResNet34
from ResNets import Resnet3D
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vit_pytorch import ViT


#2D ResNet34 + Encoder + Fc
class Transformer(ResNet34):
    def __init__(self, num_classes):
        super(Transformer,self).__init__(num_classes)
        self.transformer = TransformerModel(num=num_classes,ninp=512, nhead=4, nhid=512, nlayers=2, dropout=0.5).to('cuda:1')

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
        input = self.reshape_data(x)
        res = self.transformer(input)
        return res  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, num, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(25088, num)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) #49xbsx512
        output = output.transpose(0,1)  #bsx512
        output = output.reshape([output.size(0),25088])  # 将输出拉伸为一行：1x512
        output = self.decoder(output)
        return output


#2D ResNet34 + Encoder + Mlp
class TransformerMlp(ResNet34):
    def __init__(self, num_classes):
        super(TransformerMlp,self).__init__(num_classes)
        self.transformer = TransformerModel_MLP(num=num_classes,ninp=512, nhead=4, nhid=512, nlayers=2, dropout=0.5).to('cuda:1')

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
        input = self.reshape_data(x)
        res = self.transformer(input)
        return res  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12

class TransformerModel_MLP(nn.Module):

    def __init__(self, num, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_MLP, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Sequential(
            nn.Linear(50176,2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048,4096),
            nn.Dropout(0.2),
            nn.LayerNorm(4096),
            nn.Linear(4096,num)
        )

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) #49xbsx512
        output = output.transpose(0,1)  #bsx512
        output = output.reshape([output.size(0),50176])  # 将输出拉伸为一行：1x512
        output = self.decoder(output)
        return output



#2D ResNet34 + ViT
class ResViT(ResNet34):
    def __init__(self, num_classes):
        super(ResViT, self).__init__(num_classes)
        self.Vtrans = ViT(
    image_size = 7,
    patch_size = 1,
    num_classes = 12,
    dim = 1024,
    depth = 8,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 512
    )


    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        res = self.Vtrans(x)
        return res  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12

# ViT
class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.Vtrans = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 12,
    dim = 1024,
    depth = 8,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 3
    )


    def forward(self, x):  # 224x224x3
        res = self.Vtrans(x)
        return res  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12


#2D and 3D ResNet34 + Encoder + Mlp
class Transformer3DMLP(Resnet3D):
    def __init__(self, num_classes):
        super(Transformer3DMLP,self).__init__(num_classes)
        self.transformer = TransformerModel_MLP(num=num_classes,ninp=1024, nhead=4, nhid=1024, nlayers=2, dropout=0.5).to('cuda:1')

    def reshape_data(self,x):
        x = x.transpose(0, 2)  # 49x1024x64
        input = x.transpose(1, 2)  # 49x64x1024
        return input

    def forward(self, x):  # 224x224x3
        y = x[:,1,:,:,:]
        y = self.pre(y)  # 56x56x64
        y = self.layer1(y)  # 56x56x64
        y = self.layer2(y)  # 28x28x128
        y = self.layer3(y)  # 14x14x256
        y = self.layer4(y)  # 7x7x512
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1_3D(x)
        x = self.layer2_3D(x)
        x = self.layer3_3D(x)
        x = self.layer4_3D(x)
        x = x.reshape([x.shape[0],512,49])
        y = y.reshape([y.shape[0],512,49])
        z = torch.cat((x,y),1) #64x1024x49
        z = self.reshape_data(z)
        z = self.transformer(z)
        return z

# 2D and 3D ResNet34 + Encoder + Fc
class Transformer3D(Resnet3D):
    def __init__(self, num_classes):
        super(Transformer3D,self).__init__(num_classes)
        self.transformer = TransformerModel(num=num_classes,ninp=1024, nhead=4, nhid=1024, nlayers=2, dropout=0.5).to('cuda:1')

    def reshape_data(self,x):
        x = x.transpose(0, 2)  # 49x1024x64
        input = x.transpose(1, 2)  # 49x64x1024
        return input

    def forward(self, x):  # 224x224x3
        y = x[:,1,:,:,:]
        y = self.pre(y)  # 56x56x64
        y = self.layer1(y)  # 56x56x64
        y = self.layer2(y)  # 28x28x128
        y = self.layer3(y)  # 14x14x256
        y = self.layer4(y)  # 7x7x512
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1_3D(x)
        x = self.layer2_3D(x)
        x = self.layer3_3D(x)
        x = self.layer4_3D(x)
        x = x.reshape([x.shape[0],512,49])
        y = y.reshape([y.shape[0],512,49])
        z = torch.cat((x,y),1) #64x1024x49
        z = self.reshape_data(z)
        z = self.transformer(z)
        return z

#2D and 3D ResNet34 + ViT
class ViT3D(Resnet3D):
    def __init__(self, num_classes):
        super(Transformer3D,self).__init__(num_classes)
        self.Vtans = ViT(
    image_size = 49,
    patch_size = 1,
    num_classes = 12,
    dim = 1024,
    depth = 8,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 1024
    )

    def reshape_data(self,x):
        x = x.transpose(0, 2)  # 49x1024x64
        input = x.transpose(1, 2)  # 49x64x1024
        return input

    def forward(self, x):  # 224x224x3
        y = x[:,1,:,:,:]
        y = self.pre(y)  # 56x56x64
        y = self.layer1(y)  # 56x56x64
        y = self.layer2(y)  # 28x28x128
        y = self.layer3(y)  # 14x14x256
        y = self.layer4(y)  # 7x7x512
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1_3D(x)
        x = self.layer2_3D(x)
        x = self.layer3_3D(x)
        x = self.layer4_3D(x)
        x = x.reshape([x.shape[0],512,49])
        y = y.reshape([y.shape[0],512,49])
        z = torch.cat((x,y),1) #64x1024x49
        z = self.reshape_data(z)
        z = self.Vtans(z)
        return z