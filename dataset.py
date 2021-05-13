import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io

root_dir = "./BCICIV_2a_mat/"

class BCICIV_dataset(Dataset):
    # subject = ['S01','S02'...]
    # train = [True,False]
    # method = ['Single','X']
    def __init__(self,subject,train,method):
        self.subject = subject
        self.train = train
        self.method = method
        
    def __len__(self):
        if self.train:
            if self.method == 'Single':
                return 72*4
            elif self.method == 'X':
                return 72*4*8*2
            elif self.method == 'X_mix':
                return (72*4*8*2)+72*4
            else:
                print("check dataset method!")
                exit()
        else:
            if (self.method == 'Single') or (self.method == 'X') or (self.method == 'X_mix'):
                return 72*4
            else:
                print("check dataset method!")
                exit()

    def __getitem__(self, idx):
        # Train
        if self.train == True:
            if self.method == 'Single':
                filename = root_dir + 'BCIC_'+self.subject+'_T.mat'
                mat = scipy.io.loadmat(filename)
                signal = mat['x_train'][idx]
                label = mat['y_train'][idx]
            elif self.method == 'X':
                subject_index = idx % (72*4*2)
                subject_num = (idx // (72*4*2))
                subject_list = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
                subject_list.remove(self.subject)
                subject = subject_list[subject_num]
                T_E = subject_index // (72*4)
                if not T_E:
                    filename = root_dir + 'BCIC_'+subject+'_T.mat'
                    mat = scipy.io.loadmat(filename)
                    signal = mat['x_train'][subject_index % (72*4)]
                    label = mat['y_train'][subject_index % (72*4)]
                else:
                    filename = root_dir + 'BCIC_'+subject+'_E.mat'
                    mat = scipy.io.loadmat(filename)
                    signal = mat['x_test'][subject_index % (72*4)]
                    label = mat['y_test'][subject_index % (72*4)]
            elif self.method == 'X_mix':
                subject_index = idx % (72*4*2)
                subject_num = (idx // (72*4*2))
                subject_list = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
                subject_list.remove(self.subject)
                T_E = subject_index // (72*4)
                if not T_E:
                    # handle the indexes that are larger than subject list
                    # which indicates the subject should be self.subject
                    if subject_num == 8:
                        filename = root_dir + 'BCIC_'+self.subject+'_T.mat'
                    else:
                        subject = subject_list[subject_num]
                        filename = root_dir + 'BCIC_'+subject+'_T.mat'
                    mat = scipy.io.loadmat(filename)
                    signal = mat['x_train'][subject_index % (72*4)]
                    label = mat['y_train'][subject_index % (72*4)]
                else:
                    subject = subject_list[subject_num]
                    filename = root_dir + 'BCIC_'+subject+'_E.mat'
                    mat = scipy.io.loadmat(filename)
                    signal = mat['x_test'][subject_index % (72*4)]
                    label = mat['y_test'][subject_index % (72*4)]
        # Test
        else:
            if (self.method == 'Single') or (self.method=='X') or (self.method == 'X_mix'):
                filename = root_dir + 'BCIC_'+self.subject+'_E.mat'
                mat = scipy.io.loadmat(filename)
                signal = mat['x_test'][idx]
                label = mat['y_test'][idx]
        return np.expand_dims(signal,axis=0),label