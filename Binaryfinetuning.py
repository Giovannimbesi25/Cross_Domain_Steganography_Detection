# %% [markdown]
# # Kaggle

# %%
#https://www.youtube.com/watch?v=gkEbaMgvLs8
#https://www.kaggle.com/competitions/alaska2-image-steganalysis/data

# %%
#! kaggle competitions download -c alaska2-image-steganalysis

# %% [markdown]
# # Dataset

# %% [markdown]
# ## librerie utilizzate

# %%
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import os
from os.path import join
import sys
from collections import defaultdict
import random
import pandas as pd
import string
import timeit
import wandb
import timm




# %%
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec
class PyDataset(Dataset):
    def __init__(self,pandas_dataset,num_classes,transform=None) -> None:
        super().__init__()
        self.pandas_dataset=pandas_dataset
        self.num_classes=num_classes
        self.transform=transform
    def __len__(self):
        return len(self.pandas_dataset)
    def __getitem__(self, index) -> any:
        im_path=self.pandas_dataset.loc[index,"path"]
        image=cv2.imread(im_path)
        imageYCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        
        
        labels=self.pandas_dataset.loc[index,"class"]
        target = onehot(2, labels)
        
        if self.transform:
            image = self.transform(image)
            imageYCC = self.transform(imageYCC)
        """print("RGB:",image.size())
        print("YCC:",imageYCC.size())
        print("Stack:",torch.cat((image,imageYCC),dim=0).size())"""
        image=torch.cat((image,imageYCC),dim=0)
        return {'image': image,
                'labels': target
                }

# %% [markdown]
# Utility per la valutazione dei risultati

# %%
class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
        
    def add(self, value, num):
        self.sum += value*num
        self.num += num
        
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None
class ConfusionMatrix():
    def __init__(self) -> None:
        self.cfm=np.array([[0,0],[0,0]])
    def add(self,prediction,groundtruth):
        for i,j in zip(prediction,groundtruth):
            self.cfm[i,j]+=1
    def reset(self):
        self.cfm=np.array([[0,0],[0,0]])
    def print(self):
        print(self.cfm)

# %%
def train_classifier(model, train_loader, validation_loader, exp_name='experiment', lr=0.01, epochs=10, momentum=0.99, logdir='logs'):
    criterion = nn.CrossEntropyLoss()
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    #print("PARAMETRI:",non_frozen_parameters)
    optimizer = SGD(non_frozen_parameters, lr, momentum=momentum) 
    

    cf=ConfusionMatrix()
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    f1_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    loader = {
    'train' : train_loader,
    'validation' : validation_loader
    }


    global_step = 0
    for e in range(epochs):
        print("epoch: ",e,end='\n')
        

        for mode in ['train','validation']:
            loss_meter.reset()
            acc_meter.reset()
            precision_meter.reset()
            f1_meter.reset()
            recall_meter.reset()
            cf.reset()

            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'):
                for i, batch in enumerate(loader[mode]):
                    
                    x=batch["image"].to(device) 
                    y=batch["labels"].to(device)
                    output = model(x)
                    #print("output: ",output)
                    
                    n = x.shape[0]
                    #print(mode," iter:",i,"/",len(loader[mode]),end='\r')
                    global_step += n
                    
                    l = criterion(output,torch.max(y, 1)[1])
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    acc = accuracy_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
                    prec=precision_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1], average='weighted')
                    rec=recall_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1], average='weighted')
                    f1=f1_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1], average='weighted')

                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    precision_meter.add(prec,n)
                    recall_meter.add(rec,n)
                    f1_meter.add(f1,n)
                    #cf.add(output.to('cpu').max(1)[1],y.to('cpu').max(1)[1])
                    
                    if mode=='train':
                        wandb.log({"loss/train_bin": loss_meter.value(), 'accuracy/train_bin': acc_meter.value(),'precision/train_bin': precision_meter.value(),'recall/train_bin': recall_meter.value(),'f1/train_bin': f1_meter.value(),'custom_step_bin':e })
                        

            

            print("accuracy: ",acc_meter.value())
            print("precision: ",precision_meter.value())
            print("recall: ",recall_meter.value())
            print("F1: ",f1_meter.value())
            wandb.log({"loss/"+mode+"_bin": loss_meter.value(), 'accuracy/'+mode+"_bin": acc_meter.value(),'precision/'+mode+"_bin": precision_meter.value(),'recall/'+mode+"_bin": recall_meter.value(),'f1/'+mode+"_bin": f1_meter.value(),'custom_step'+"_bin":e })
            

        
        
        
        if (e+1)%5==0:
            torch.save(model.state_dict(),'%s-%d.pth'%(exp_name,e+1))
    return model

# %%
def test_classifier(model, test_loader, exp_name='experiment',test_name="test", logdir='logs'):
    predictions=[]
    criterion = nn.CrossEntropyLoss()


    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    f1_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()
    cf=ConfusionMatrix()
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_meter.reset()
    acc_meter.reset()
    model.eval()
    global_step=0
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(test_loader):
            
            x=batch["image"].to(device) 
            y=batch["labels"].to(device)
            output = model(x)
            predictions.append(output)
            
            n = x.shape[0] 
            print("Test"," iter:",i,"/",len(test_loader),end='\r')
            global_step += n
            """
            l = criterion(output,torch.max(y, 1)[1])
            acc = accuracy_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            prec=precision_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            rec=recall_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            f1=f1_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            loss_meter.add(l.item(),n)
            acc_meter.add(acc,n)
            precision_meter.add(prec,n)
            recall_meter.add(rec,n)
            f1_meter.add(f1,n)
            cf.add(output.to('cpu').max(1)[1],y.to('cpu').max(1)[1])
        wandb.log({"loss/"+test_name: loss_meter.value(), 'accuracy/'+test_name: acc_meter.value(),'precision/'+test_name: precision_meter.value(),'recall/'+test_name: recall_meter.value(),'f1/'+test_name: f1_meter.value() })
         """
        """writer.add_scalar('loss/' +test_type, loss_meter.value(), global_step=global_step)
        writer.add_scalar('accuracy/' + test_type, acc_meter.value(), global_step=global_step)
        writer.add_scalar('precision/' + test_type, precision_meter.value(), global_step=global_step)
        writer.add_scalar('recall/' + test_type, recall_meter.value(), global_step=global_step)
        writer.add_scalar('f1/' + test_type, f1_meter.value(), global_step=global_step)"""
        print("Test Loss: ",loss_meter.value())
        print("Test Accuracy: ",acc_meter.value())
    return predictions

# %%
def test_alaska(model,test_path="/home/rforte/MultimediaProject/MultimediaProject/Test"):
    test_path="/home/rforte/MultimediaProject/MultimediaProject/SUNIWARD"
    files=os.listdir(test_path)

    df = pd.DataFrame(columns=["Id","Label"])
    print(type(df))
    device="cuda"
    #print(files)
    pred=0
    for file in files:
        image=cv2.imread(os.path.join(test_path,file))
        image=transforms.ToTensor()(image).unsqueeze(0).to(device)
        #print(type(image))
        model.to(device)
        model.eval()
        out=model(image).to('cpu').max(1)[1]
        if out>=1:
            pred=pred+1
        print("ACCURACY S-UNIWARD: ")
        print(pred/len(files))
        #row={"Id":pred,"Label":file}
        #df.loc[len(df)] = [file,pred]
    
    #df.to_csv("alaska_pred.csv",index=False)

# %%
def build_df():
    train_stego_dir=['Cover','JMiPOD','JUNIWARD','UERD']
    record=[]
    feature_list={}
    label=0
    for dir in train_stego_dir:
        l=os.listdir(dir)
        for im in l:
            feature_list["path"]=os.path.join(dir,im)
            feature_list["class"]=label
            record.append(feature_list.copy())
        label=1

    df = pd.DataFrame.from_records(record)
    return df
def split_df(df,split_factor,seed):
    df = df.sample(frac=1,random_state=seed).reset_index(drop=True)
    df_train, df_val =train_test_split(df,test_size=split_factor,random_state=seed)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    return df_train, df_val

# %%
def get_models():
    models_list=[]
    efficentnet = models.efficientnet_b2(weights=True)
    efficentnet._fc = nn.Linear(in_features=1408, out_features=2, bias=True)
    
    #efficentnet= nn.Sequential(efficentnet,nn.Linear(in_features=1000, out_features=2, bias=True))
    #efficentnet[0].features[0][0]=nn.Conv2d(6, 32, kernel_size=3, stride=2, bias=False)
    efficentnet.load_state_dict(torch.load("/home/rforte/MultimediaProject/MultimediaProject/efficientnet_b2_Alaska2_BINARY0.001-bin-5.pth"))

    
    models_list.append({'name': 'efficientnet_b2', 'model': efficentnet, 'batch_size': 16 })

    """resnet18=timm.create_model('resnet18', pretrained=True)
    resnet18 = nn.Sequential(resnet18, nn.Linear(1000, 4), nn.Softmax(dim=1))
    models.append({'name': 'resnet18', 'model': resnet18, 'batch_size': 64})

    resnet50=timm.create_model('resnet50', pretrained=True)
    resnet50 = nn.Sequential(resnet50, nn.Linear(1000, 4), nn.Softmax(dim=1))
    models.append({'name': 'resnet50', 'model': resnet50, 'batch_size': 16})

    resnet101=timm.create_model('resnet101', pretrained=True)
    resnet101 = nn.Sequential(resnet101, nn.Linear(1000, 4), nn.Softmax(dim=1))
    models.append({'name': 'resnet101', 'model': resnet101, 'batch_size': 8})

    densenet121=timm.create_model('densenet121', pretrained=True)
    densenet121 = nn.Sequential(densenet121, nn.Linear(1000, 4), nn.Softmax(dim=1))
    models.append({'name': 'densenet121', 'model': densenet121, 'batch_size': 16})

    maxvit_tiny=timm.create_model('maxvit_tiny_tf_512', pretrained=True)
    maxvit_tiny = nn.Sequential(maxvit_tiny, nn.Linear(1000, 4), nn.Softmax(dim=1))
    models.append({'name': 'maxvit_tiny__tf_512', 'model': maxvit_tiny, 'batch_size': 4})
    """
    return models_list

# %%
def get_loader(df_train,df_val,transform,num_classes,batch_size):
    train_dataset = PyDataset(df_train,
                                   num_classes=num_classes,
                                   transform=transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    validation_dataset = PyDataset(df_val,
                                    num_classes=num_classes,
                                    transform=transform)
    validation_dataset_loader=torch.utils.data.DataLoader(validation_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    return train_dataset_loader, validation_dataset_loader

# %%
def log_init(exp_name, model_name,lr, batch_size,dataset_name,epochs):
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="MultimediaProject",
        name=exp_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "batch_size":batch_size,
            "architecture": model_name,
            "dataset": dataset_name,
            "epochs": epochs,
        })

# %%
NUM_CLASSES = 4
device = torch.device("cuda")
seed=42
split_factor=0.25
epochs=30

df=build_df()

train_df, val_df=split_df(df,split_factor,seed)
transform = transforms.Compose([
    transforms.ToTensor()
])
models_list=get_models()

#print(train_df.head(100))


# %%

for model in models_list:
    for lr in [ 0.001]:
        train_dataset_loader, validation_dataset_loader=get_loader(train_df,val_df,transform,2,model["batch_size"])
        
        exp_name=model["name"]+"_Alaska2_BINARY_RGB"+str(lr)
        log_init(exp_name,model["name"],lr,model["batch_size"],"Alaska2",epochs)
        """params_to_train=["_fc.weight","_fc.bias"]
        for name,param in model["model"].named_parameters():
            param.requires_grad = True if name in params_to_train else False"""
        
        #trained_model = train_classifier(model["model"], train_dataset_loader, validation_dataset_loader,exp_name, epochs = epochs,lr=lr)
        predictions= test_alaska(model["model"])
        wandb.finish()

# %%



