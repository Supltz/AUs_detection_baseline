import os
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from tqdm import tqdm



def get_sequences_task():
    sequences = []
    task = []
    for i in range(23):
        sequences.append('F' + str(i+1).zfill(3))
    for i in range(18):
        sequences.append('M' + str(i+1).zfill(3))
    for i in range(8):
        task.append('T' + str(i+1))
    return sequences, task


# def get_AUlabels(path):
#     usecols = ['0', '1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
#     df = pd.read_csv(path, header=0, index_col=0, usecols=usecols)
#     frames = [str(item) for item in list(df.index.values)]
#     frames.sort()
#     labels = df.values
#     # 返回的frames是list，值是排好序的int变量，指示对应的帧。labels是N*12的np.ndarray，对应AU标签
#     return frames, labels


# def get_images(folder_path, frames):
#     images = []
#     for frame in frames:
#         path = os.path.join(folder_path, 'frame_det_00_{}.bmp'.format(frame.zfill(6)))
#         if os.path.exists(path):
#             temp = Image.open(path)
#             if temp.getbands()[0] == 'L':
#                 temp = temp.convert('RGB')
#             temp_np = np.array(temp)    #这里我不加reshape(3,240,240)？
#             temp.close()
#             images.append(temp_np)
#     # 返回的images是list类型
#     return images


# def save_pkl():
#     # save pkl
#     label_folder = os.path.join('/data', 'wenqi', 'datasets', 'BP4D', 'AUCoding', 'AU_Extract')
#     Detect_folder = os.path.join('/data', 'wenqi', 'datasets', 'BP4D', 'Face_Detect')
#     pkl_folder = os.path.join('/data', 'wenqi', 'datasets', 'BP4D', 'PKL')
#     sequences, task = get_sequences_task()
#     for seq in sequences:
#         print('loading the sequences: ', seq)
#         for t in task:
#             path_label = os.path.join(label_folder, '{sequence}_{task}.csv'.format(sequence=seq, task=t)) #读取AU标签路径
#             temp_frames, temp_labels = get_AUlabels(path_label)
#             image_folder = os.path.join(Detect_folder, seq, t, '{}_aligned'.format(t))
#             temp_images = get_images(image_folder, frames=temp_frames)
#             # 读取了之后进行整合
#             if t == 'T1':
#                 labels = temp_labels
#                 images = temp_images
#             else:
#                 labels = np.concatenate((labels, temp_labels), axis=0)  #np.ndarray
#                 images = images + temp_images   #list
#         if not os.path.exists(pkl_folder):
#             os.mkdir(pkl_folder)
#         print('dumping the sequence {}'.format(seq))
#         print(len(images), labels.shape)
#         pickle.dump((images, labels), open(os.path.join(pkl_folder, '{}.pkl'.format(seq)), 'wb'))
#         print('the sequence {} has been dumped'.format(seq))

def load_data(sequences):
    # 用于读取pkl中的数据，并将不同序列的数据整合到一起。用于获得train和validation等
    import main_bp4d
    pkl_folder = os.path.join(main_bp4d.args.PATH_dataset)
    type = main_bp4d.args.datatype
    for seq in sequences:
        print('loading the pkl: {}.pkl'.format(seq))
        temp_images, temp_labels = pickle.load(open(os.path.join(pkl_folder, '{}.pkl'.format(seq)), 'rb'))
        if seq == sequences[0]:
            images = temp_images
            labels = temp_labels
        else:
            images = images + temp_images
            labels = np.concatenate((labels, temp_labels), axis=0)
    print("Erase the dirty data")
    i=0
    pbar = tqdm(i=len(labels)-1)
    while(i!=len(labels)):
        if(sum(labels[i,:])==0):
            labels = np.delete(labels, i, axis=0)
            images.pop(i)
        else:
            i=i+1
            pbar.update(1)
    pbar.close()
    # 返回images是list，单个值对应单张图片,shape(240,240,3)。labels是numpy.ndarray
    #返回值要有三种选择 一种返回11W张 一种返回8W张3各一组 一种返回8W张分开的  验证集也是这三种选择  还要进行Transpose 变成（N,C,T,H,W）
    if(type == "static"):
        return images, labels #train:124048   删除后：113149    validation删除后:22079
    elif(type == "dynamic"):
        j = 10
        D_frames = []
        D_labels = []
        print("Preparing the dynamic data:")
        pbar = tqdm(j=len(labels) - 11)
        while (j != len(labels) - 10):
            if (any(labels[j] != labels[j - 10]) and any(labels[j] != labels[j + 10])):
                frames = np.concatenate([np.expand_dims(images[j - 10], axis=0), np.expand_dims(images[j], axis=0),
                                         np.expand_dims(images[j + 10], axis=0)], axis=0)
                D_frames.append(frames)
                D_labels.append(labels[j])
                j = j + 1
                pbar.update(1)
            else:
                j = j + 1
                pbar.update(1)
        pbar.close()
        return D_frames,D_labels
    elif(type == "dynamic_one_frame"):
        j = 10
        D_one_frame = []
        D_one_labels = []
        print("Preparing the dynamic data(in single frame):")
        pbar = tqdm(j=len(labels) - 11)
        while (j != len(labels) - 10):
            if (any(labels[j] != labels[j - 10]) and any(labels[j] != labels[j + 10])):
                D_one_frame.append(images[j - 10])
                D_one_labels.append(labels[j - 10])
                D_one_frame.append(images[j])
                D_one_labels.append(labels[j])
                D_one_frame.append(images[j + 10])
                D_one_labels.append(labels[j + 10])
                j = j + 1
                pbar.update(1)
            else:
                j = j + 1
                pbar.update(1)
        pbar.close()
        return D_one_frame,D_one_labels
