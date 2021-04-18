import os
import numpy as np
import pickle

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
    i=0
    while(i!=len(labels)):
        if(sum(labels[i,:])==0):
            labels = np.delete(labels, i, axis=0)
            images.pop(i)
        else:
            i=i+1
    # 返回images是list，单个值对应单张图片,shape(240,240,3)。labels是numpy.ndarray
    if(type == "static"):
        return images, labels #train:124048   删除后：113149    validation删除后:22079
    elif(type == "dynamic"):
        j = 10
        D_frames = []
        D_labels = []
        while (j != len(labels) - 10):
            if (any(labels[j] != labels[j - 10]) and any(labels[j] != labels[j + 10])):
                frames = np.concatenate([np.expand_dims(images[j - 10], axis=0), np.expand_dims(images[j], axis=0),
                                         np.expand_dims(images[j + 10], axis=0)], axis=0)
                D_frames.append(frames)
                D_labels.append(labels[j])
                j = j + 1
            else:
                j = j + 1
        return D_frames,D_labels
    elif(type == "dynamic_in_frames"):
        j = 10
        D_one_frame = []
        D_one_labels = []
        while (j != len(labels) - 10):
            if (any(labels[j] != labels[j - 10]) and any(labels[j] != labels[j + 10])):
                D_one_frame.append(images[j - 10])
                D_one_labels.append(labels[j - 10])
                D_one_frame.append(images[j])
                D_one_labels.append(labels[j])
                D_one_frame.append(images[j + 10])
                D_one_labels.append(labels[j + 10])
                j = j + 1
            else:
                j = j + 1
        return D_one_frame,D_one_labels
