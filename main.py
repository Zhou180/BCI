import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import time

import model
from dataset import BCICIV_dataset
from model.model_pytorch import EEGNet,ShallowConvNet,SCCNet,TSception

def train(net, train_loader, test_loader, optimizer, criterion, device, args):
    step = 0
    print('Training...')
    exp_pbar = tqdm(range(args.epoch))

    for epoch in exp_pbar:
        net.train()
        running_loss = 0
        batch_idx = 0
        epoch_pbar = tqdm(train_loader)
        for signals,labels in epoch_pbar:
            signals,labels = signals.to(device).to(torch.float), labels.to(device).to(torch.long).squeeze(1)
            optimizer.zero_grad()
            output = net(signals)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"step_loss": loss.item(),'step':step})
            step += 1
            epoch_pbar.set_description(f'epoch: {epoch:>2}/{args.epoch}, batch: {batch_idx:>3}/{len(train_loader)}, loss: {loss.item():.5f}')
            batch_idx += 1
        
        running_loss /= batch_idx
        wandb.log({"epoch_loss": running_loss,'Epoch':epoch})
        print('----------------------------')
        exp_pbar.set_description(f'epoch: {epoch:>2}/{args.epoch}, avg. loss: {running_loss:.5f}')
        print('----------------------------')
        evaluate(net, test_loader, optimizer, criterion, device, args, epoch)

def train_with_finetune(net, train_loader, finetune_loader, test_loader, optimizer, criterion, device, args):
    step = 0
    print('Training...')
    exp_pbar = tqdm(range(args.epoch))

    for epoch in exp_pbar:
        net.train()
        running_loss = 0
        batch_idx = 0
        epoch_pbar = tqdm(train_loader)
        for signals,labels in epoch_pbar:
            signals,labels = signals.to(device).to(torch.float), labels.to(device).to(torch.long).squeeze(1)
            optimizer.zero_grad()
            output = net(signals)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"step_loss": loss.item(),'step':step})
            step += 1
            epoch_pbar.set_description(f'epoch: {epoch:>2}/{args.epoch}, batch: {batch_idx:>3}/{len(train_loader)}, loss: {loss.item():.5f}')
            batch_idx += 1
        
        running_loss /= batch_idx
        wandb.log({"epoch_loss": running_loss,'Epoch':epoch})
        print('----------------------------')
        exp_pbar.set_description(f'epoch: {epoch:>2}/{args.epoch}, avg. loss: {running_loss:.5f}')
        print('----------------------------')
        evaluate(net, test_loader, optimizer, criterion, device, args, epoch)
    
    print('Finetuning...')
    finetune_exp_pbar = tqdm(range(args.finetune_epoch))
    for epoch in finetune_exp_pbar:
        net.train()
        running_loss = 0
        batch_idx = 0
        finetune_epoch_pbar = tqdm(finetune_loader)
        for signals,labels in finetune_epoch_pbar:
            signals,labels = signals.to(device).to(torch.float), labels.to(device).to(torch.long).squeeze(1)
            optimizer.zero_grad()
            output = net(signals)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"step_loss": loss.item(),'step':step})
            step += 1
            finetune_epoch_pbar.set_description(f'epoch: {epoch:>2}/{args.finetune_epoch}, batch: {batch_idx:>3}/{len(finetune_loader)}, loss: {loss.item():.5f}')
            batch_idx += 1
        
        running_loss /= batch_idx
        wandb.log({"epoch_loss": running_loss,'Epoch':epoch+args.epoch})
        print('----------------------------')
        finetune_exp_pbar.set_description(f'epoch: {epoch:>2}/{args.finetune_epoch}, avg. loss: {running_loss:.5f}')
        print('----------------------------')
        evaluate(net, test_loader, optimizer, criterion, device, args, epoch+args.epoch)

def evaluate(net, test_loader, optimizer, criterion, device, args, epoch):
    global best_test_loss
    net.eval()
    print('============================')
    net.eval()
    print("Evaluating..")
    print('Test Set:')
    test_pbar = tqdm(test_loader)
    batch_idx = 0
    test_loss = 0
    result = []
    grnd_t = []
    for signals,labels in test_pbar:
        signals,labels = signals.to(device).to(torch.float), labels.to(device).to(torch.long).squeeze(1)
        with torch.no_grad():
            output = net(signals)
            loss = criterion(output,labels)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            result = np.append(result,pred.squeeze().cpu().numpy())
            grnd_t = np.append(grnd_t,labels.squeeze().cpu().numpy())
        batch_idx += 1
    test_loss = test_loss/batch_idx
    print("")
    print("Test Loss:", test_loss)

    wandb.log({"Test Loss": test_loss,'Epoch':epoch})
    cm = confusion_matrix(grnd_t,result)
    class_acc = confusion_matrix(grnd_t,result,normalize="true").diagonal()
    total_acc = class_acc.sum() / 4

    print("Confusion Matrix:")
    print(cm)
    print("Class Accuracy:")
    print(class_acc)
    print("Accuracy:")
    print(total_acc)

    wandb.log({ "conf_mat" : wandb.plot.confusion_matrix(probs=None
                            ,y_true=grnd_t, preds=result),
                'Epoch':epoch})
    wandb.log({"Class Accuracy": class_acc, 'Epoch':epoch})
    wandb.log({"Accuracy": total_acc,'Epoch':epoch})

    print('============================')

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default=None,help='Run name to display on wandb')
    parser.add_argument('--subject', type=str,
                        help='Subject for training in within subject, or Subject to exlude in X subject')
    parser.add_argument('--method',type=str,
                        help='The method of training[Single, X, X_finetune]')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--finetune_epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32,required=True)
    parser.add_argument('--learning_rate',type=float,default=1e-3,required=True)
    parser.add_argument('--net', type=str, required=True,
                        help='The network [EEGNet,ShallowConvNet,SCCNet,TSception]')
    parser.add_argument('--shuffle',type=str,default='True',required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--tags',type=str,nargs='*')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = load_args()
    print(args)
    
    if args.gpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif args.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.method=='Single':
        train_set = BCICIV_dataset(args.subject,True,"Single")
        train_loader = DataLoader(train_set,args.batch,shuffle=args.shuffle)
        test_set = BCICIV_dataset(args.subject,False,"Single")
        test_loader = DataLoader(test_set,args.batch,shuffle=args.shuffle)
    elif (args.method=='X') or (args.method=='X_finetune'):
        train_set = BCICIV_dataset(args.subject,True,"X")
        train_loader = DataLoader(train_set,args.batch,shuffle=args.shuffle)
        test_set = BCICIV_dataset(args.subject,False,"X")
        test_loader = DataLoader(test_set,args.batch,shuffle=args.shuffle)
    elif args.method=='X_mix':
        train_set = BCICIV_dataset(args.subject,True,"X_mix")
        train_loader = DataLoader(train_set,args.batch,shuffle=args.shuffle)
        test_set = BCICIV_dataset(args.subject,False,"X_mix")
        test_loader = DataLoader(test_set,args.batch,shuffle=args.shuffle)

    if args.method=='X_finetune':
        finetune_set =  BCICIV_dataset(args.subject,True,"Single")
        finetune_loader = DataLoader(finetune_set,args.batch,shuffle=args.shuffle)
    
    if args.net == "TSception":
        net = TSception(4, (1,22,562) ,250, 9, 6, 128, 0.2).to(device)
    else:
        net = eval(args.net+"().to(device)")

    optimizer = optim.Adam(net.parameters(),lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    wandb.init(project="BCI_Lab4",name=args.name,tags=args.tags,save_code=True)
    wandb.watch(net)
    if args.method=='X_finetune':
        train_with_finetune(net, train_loader, finetune_loader, test_loader, optimizer, criterion, device, args)
    else:
        train(net, train_loader, test_loader, optimizer, criterion, device, args)

