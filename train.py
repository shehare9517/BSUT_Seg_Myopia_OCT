#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import *
from model.TransMUNet import TransMUNet
from network import U_Net
# from model.resnet_utnet import ResNet_UTNet
# from model.res_utnet_AG import ResNet_UTNet
from model.Bitrans_DAG_7 import Bi_TransNet
import pandas as pd
import glob
import nibabel as nib
import numpy as np
import copy
import yaml

# In[2]:


## Loader
## Hyper parameters
config = yaml.load(open('./config_attunet.yml'), Loader=yaml.FullLoader)
save_path = './weights/BiTrans_DAG71/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Net = Bi_TransNet(in_ch=3, num_class=1)

Net = Net.to(device)

data_path = config['path_to_data']

train_dataset = isic_loader(path_Data=data_path, train=True)
train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size_tr']), shuffle=True)
val_dataset = isic_loader(path_Data=data_path, train=False)
val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size_va']), shuffle=False)

# In[3]:


# Net = TransMUNet(n_classes=number_classes)
# Net = U_Net()

if int(config['pretrained']):
    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']
optimizer = optim.Adam(Net.parameters(), lr=float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config['patience'])
criteria = torch.nn.BCELoss()
criteria_boundary = torch.nn.BCELoss()
criteria_region = torch.nn.MSELoss()

# In[ ]:


for ep in range(int(config['epochs'])):
    Net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device)
        weak_ann = batch['weak_ann'].to(device)
        boundary = batch['boundary'].to(device)
        # mask_type = torch.float32 if Net.n_classes == 1 else torch.long
        mask_type = torch.float32
        msk = msk.to(device=device, dtype=mask_type)
        weak_ann = weak_ann.to(device=device, dtype=mask_type)
        boundary = boundary.to(device=device, dtype=mask_type)
        msk_pred, Bound = Net(img)
        loss = criteria(msk_pred, msk)
        # loss_regions = criteria_region(weak_ann[:, 0], R[:, :-1, 0])
        loss_boundary = criteria(Bound, boundary)
        # tloss = (0.7 * (loss)) + (0.1 * loss_regions) + (0.2 * loss_boundary)
        tloss = 0.8 * loss + 0.2 * loss_boundary
        optimizer.zero_grad()
        tloss.backward()
        epoch_loss += tloss.item()
        optimizer.step()
        if itter % int(float(config['progress_p']) * len(train_loader)) == 0:
            print(f' Epoch>> {ep + 1} and itteration {itter + 1} Loss>> {((epoch_loss / (itter + 1)))}')
    ## Validation phase
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, batch in enumerate(val_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            # mask_type = torch.float32 if Net.n_classes == 1 else torch.long
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred, _ = Net(img)
            loss = criteria(msk_pred, msk)
            val_loss += loss.item()
        print(f' validation on epoch>> {ep + 1} dice loss>> {(abs(val_loss / (itter + 1)))}')
        mean_val_loss = (val_loss / (itter + 1))

        if (mean_val_loss) < best_val_loss:
            print('New best loss, saving...')
            best_val_loss = copy.deepcopy(mean_val_loss)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
            # torch.save(state, config['saved_model'])
            torch.save(state, os.path.join(save_path, 'Bestmodel_%d') % (ep))
        elif ep % 10 == 0:
            print('Saving model...')
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, os.path.join(save_path, 'epoch_%d') % (ep))

    scheduler.step(mean_val_loss)

print('Trainng phase finished')


