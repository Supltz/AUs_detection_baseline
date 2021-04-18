from torch.utils.data import  DataLoader
import torch.optim as optim
from tqdm import tqdm
from L_Net import *
from Transformers import *
from MyDatasets import *
import torch.nn.functional as F
from torchvision import transforms
import warnings
import argparse
from torch.utils.tensorboard import SummaryWriter

#args and settings
warnings.filterwarnings("ignore")
writer1 = SummaryWriter('runs_20')
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--device', default="cuda:1", type=str)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--N_fold', default=6, type=int, help="the ratio of train and validation data")
parser.add_argument('--PATH_Checkpoint', default="./checkpoint/CHECKPOINT_FILE", type=str)
parser.add_argument('--PATH_pretrain', default="./Resnet34model/model_state.pth", type=str)
parser.add_argument('--PATH_dataset', default="./", type=str)
parser.add_argument('--datatype', default=None, type=str, help="choose from dynamic dynamic_in_frames static")
parser.add_argument('--dataset', default="BP4D", type=str)
parser.add_argument('--FirstTimeRunning', default=None, type=str, help="No means reading from the checkpoint_file")
parser.add_argument('--model', default=None, type=str, help="choose from ResNet34,L_Net,Transformer,Vit,ResViT,TransformerMLP,ResNet3D,Transformer3D,Transformer3DMlp,ViT3D")
args = parser.parse_args()

# weight
def weighted(dataloader):
    total_in_epoch = [0 for i in range(len(au_keys))]
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        total = []
        for i in range(len(au_keys)):
            total.append(targets[:, i].sum().item())
        for i in range(len(au_keys)):
            total_in_epoch[i] = total[i] + total_in_epoch[i]
    Alldata=batchsize*batch_idx+len(inputs)
    weight=[]
    for i in range(len(au_keys)):
        weight.append((Alldata-total_in_epoch[i])/total_in_epoch[i])
    weight_to_tensor=torch.FloatTensor(weight)
    return weight_to_tensor


def Get_TPs(outputs, targets,TPs_in_allset,TNs_in_allset,FNs_in_allset,FPs_in_allset):
    m = nn.Sigmoid()
    predicted = m(outputs)
    AU_TP= []
    AU_TN = []
    AU_FN = []
    AU_FP = []
    for i in range(len(predicted)):
        for j in range(len(au_keys)):
            if(predicted[i,j].item()<0.5):
                predicted[i,j]=0
            else:
                predicted[i,j]=1
    for i in range(len(au_keys)):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for j in range(len(predicted)):
            if (predicted[j, i].item() == 1 and targets[j, i].item() == 1):
                TP = TP+1
            if (predicted[j, i].item() == 0 and targets[j, i].item() == 0):
                TN = TN+1
            if (predicted[j, i].item() == 0 and targets[j, i].item() == 1):
                FN = FN+1
            if (predicted[j, i].item() == 1 and targets[j, i].item() == 0):
                FP = FP+1
        AU_TP.append(TP)
        AU_TN.append(TN)
        AU_FN.append(FN)
        AU_FP.append(FP)
    for i in range(len(au_keys)):
        TPs_in_allset = AU_TP[i] + TPs_in_allset[i]
        TNs_in_allset = AU_TN[i] + TNs_in_allset[i]
        FNs_in_allset = AU_FN[i] + FNs_in_allset[i]
        FPs_in_allset = AU_FP[i] + FPs_in_allset[i]


def calculate_Acc(outputs,targets,acc_sum_allset,total_sum_allset,acc_in_allset,au_keys):
    sum_pred = [0 for i in range(len(au_keys))]  # 一个batch的准确率之和
    m = nn.Sigmoid()
    predicted = m(outputs)
    for i in range(len(targets)):
        if (targets[i, :].nonzero().numel() != 0):
            index_1 = targets[i, :].nonzero()
            for j in index_1:
                sum_pred[j] = sum_pred[j] + predicted[i, j].item()
    sum_inbatch = []
    acc_inbatch = []
    Sum = 0
    Not_nan = 0
    for i in range(len(au_keys)):
        sum_inbatch.append(targets[:, i].sum().item())
        if (sum_inbatch[i] != 0):
            acc = sum_pred[i] / sum_inbatch[i]
            acc_inbatch.append(acc)
        else:
            acc = 0.5
            acc_inbatch.append(acc)
        if (acc_inbatch[i] != 0):
            Sum = acc_inbatch[i] + Sum
            Not_nan = Not_nan + 1
    if (Not_nan == 0):
        print("This batch contains no AUs")  #load_data的时候已经把全0的去掉了，这里还是象征性的判断一下
    else:
        # avg_acc = Sum / Not_nan                  #一个batch得平均准确率 和loss
        # print(str(batch_idx) + 'avg:{:.6f}|loss:{:.6f}'.format(avg_acc, loss.item())) # 没什么意义，如果想看就把参数传进来
        for i in range(len(au_keys)):
            acc_sum_allset[i] = sum_pred[i] + acc_sum_allset[i]
            total_sum_allset[i] = sum_inbatch[i] + total_sum_allset[i]
            if (total_sum_allset[i] != 0):
                acc_in_allset[i] = acc_sum_allset[i] / total_sum_allset[i]
            else:
                acc_in_allset[i] = 0
    return acc_in_allset

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # net.module
    net.train()
    train_loss = 0
    acc_sum_allset = [0 for i in range(len(au_keys))]   #整个set的概率和
    total_sum_allset = [0 for i in range(len(au_keys))] #整个set的AU数量
    acc_in_allset = [0 for i in range(len(au_keys))] #整个set的准确率
    weight = weighted(trainloader).to(device)
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device).squeeze(dim=1).float()
        optimizer.zero_grad()
        # inputs.type/shape:torch.Tensor/[128,3,224,224]
        # targets的shape是batch_size * 12
        outputs = net(inputs)
        # functional.BCE_with_logits自范带对预测分数进行sigmoid操作，因此可以无所谓outputs的取值
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean',pos_weight=weight).to(device)
        # flood = (loss - 0.6).abs() + 0.6
        loss.backward()    #loss flooding?
        optimizer.step()
        train_loss += loss.item()
        train_acc_in_allset=calculate_Acc(outputs,targets,acc_sum_allset,total_sum_allset,acc_in_allset,au_keys)
    Aus_dict={}
    for i in range(len(au_keys)):
        Aus_dict.update({au_keys[i]:train_acc_in_allset[i]})
    print(Aus_dict)
    train_loss = train_loss / (batch_idx+1)
    train_acc=sum(train_acc_in_allset)/len(train_acc_in_allset)
    print(train_acc)
    return train_loss,train_acc,Aus_dict

@torch.no_grad()
def val(epoch):
    print('\nEpoch(validation): %d' % epoch)
    net.eval()
    val_loss = 0
    TPs_in_allset = [0 for i in range(len(au_keys))]  # 整个set的TPs
    TNs_in_allset = [0 for i in range(len(au_keys))]
    FNs_in_allset = [0 for i in range(len(au_keys))]
    FPs_in_allset = [0 for i in range(len(au_keys))]
    weight=weighted(valloader).to(device)
    for batch_idx, (inputs, targets) in enumerate(tqdm(valloader)):
        inputs, targets = inputs.to(device), targets.to(device).squeeze(dim=1).float()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean',pos_weight=weight).to(device)
        val_loss=val_loss+loss.item()
        Get_TPs(outputs,targets,TPs_in_allset,TNs_in_allset,FNs_in_allset,FPs_in_allset)
    p = []
    r = []
    F1 = []
    acc = []
    for i in range(len(au_keys)):
        p.append(TPs_in_allset[i] / (TPs_in_allset[i] + FPs_in_allset[i]))
        r.append(TPs_in_allset[i] / (TPs_in_allset[i] + FNs_in_allset[i]))
        F1.append(2 * r[i] * p[i] / (r[i] + p[i]))
        acc.append((TPs_in_allset[i] + TNs_in_allset[i]) / (TPs_in_allset[i] + TNs_in_allset[i] + FPs_in_allset[i] + FNs_in_allset[i]))
    Aus_acc_dict = {}
    Aus_F1_dict = {}
    for i in range(len(au_keys)):
        Aus_acc_dict.update({au_keys[i]: acc[i]})
        Aus_F1_dict.update({au_keys[i]: F1[i]})
    print("Accuracy:"+Aus_acc_dict)
    print("F1_Score:"+Aus_F1_dict)
    val_loss = val_loss / (batch_idx+1)
    val_acc = sum(acc) / len(acc)
    val_F1 = sum(F1) / len(F1)
    print("Avg_Acc:"+val_acc)
    print("Avg_F1"+val_F1)
    return val_loss, val_acc,val_F1,Aus_acc_dict,Aus_F1_dict

def main():
    for epoch in range(start_epoch + 1, num_epoch):
        loss_dict = {}
        acc_dict = {}
        tra_loss,tra_acc,Aus_in_tra = train(epoch)
        val_loss,val_acc,val_F1,Aus_in_val,Aus_F1 = val(epoch)
        loss_dict.update({'train_loss': tra_loss})
        loss_dict.update({'val_loss': val_loss})
        acc_dict.update({'train_acc': tra_acc})
        acc_dict.update({'val_acc': val_acc})
        acc_dict.update({'val_F1': val_F1})
        writer1.add_scalars('loss', loss_dict, global_step=epoch)
        writer1.add_scalars('acc', acc_dict, global_step=epoch)
        writer1.add_scalars('Aus_in_train', Aus_in_tra, global_step=epoch)
        writer1.add_scalars('Aus_in_val', Aus_in_val, global_step=epoch)
        writer1.add_scalars('Aus_in_val_F1', Aus_F1, global_step=epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, args.PATH_Checkpoint)

def data_augmentation():
    print('==> Preparing data...')
    sequences, _ = BP4D_load_data.get_sequences_task()   # sequences, _ filenames
    train_seq, val_seq = get_train_val(sequences)   #MyDatasets.get_train_val()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_seq,val_seq,transform_train,transform_val

# ---------------------------------------------------------------------------------
if __name__=="__main__":
    # parameters
    yes_no = args.FirstTimeRunning
    batchsize = args.batchsize
    train_lr = args.lr
    start_epoch = -1  # start from epoch 0 or last checkpoint epoch
    num_epoch = args.num_epoch
    au_keys = ['au1', 'au2', 'au4', 'au6', 'au7', 'au10', 'au12', 'au14', 'au15', 'au17', 'au23', 'au24']

    #Datalaoder
    train_seq,val_seq,transform_train,transform_val=data_augmentation()
    print("The train set:")
    trainset = MyBP4D(train_seq, train=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers)
    print("The validation set:")
    valset = MyBP4D(val_seq, train=False, transform=transform_val)
    valloader = DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)
    #check CUDA
    if torch.cuda.is_available():
        deviceidx = [0, 1]
        device= args.device
    else:
        device='cpu'

    #Models
    print("start the net")
    Nets = {"ResNet34":ResNet34(12),"L_Net":L_Net(12),"Transformer":Transformer(12),"Vit":Vit(12),
            "ResViT":ResViT(12),"TransformerMLP":TransformerMlp(12),"ResNet3D":Resnet3D(12),
            "Transformer3D":Transformer3D(12),"Transformer3DMlp":Transformer3DMLP(12),"ViT3D":ViT3D(12)}
    net = Nets[args.model]
    # if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
    #     net = torch.nn.DataParallel(net,deviceidx)    #多gpu训练,自动选择gpu
    net.to(device)

    #optimizer
    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=0.9, weight_decay=5e-4)
    print('the learning rate is ', train_lr)

    #断点重传
    if(yes_no=="no"):
        print('-----------------------------')
        path_checkpoint =args.PATH_Checkpoint
        checkpoint = torch.load(path_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("start_epoch:", start_epoch)
        print('-----------------------------')
        main()
    if(yes_no=="yes"):
        path_pretrain = args.PATH_pretrain
        pretrained = torch.load(path_pretrain)
        model_dict = net.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        net.load_state_dict(model_dict)
        main()




#验证集需不需要挑动态数据




