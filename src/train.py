# train.py 

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from timm.utils.model_ema import ModelEmaV2
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
import torch.nn.functional as F






# dataset generator
from dataset import SkinCancerDatasetRetriever

# model
from models import EfficientNetB7ClsHead

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm


# Parameters 						---------	---------	---------
classes = [
    'Melanoma',
    'Melanocytic nevus',
    'Basal cell carcinoma',
    'Actinic keratosis',
    'Benign keratosis', # Also: (solar lentigo / seborrheic keratosis / lichen planus-like keratosis).
    'Dermatofibroma',
    'Vascular lesion',
    'Squamous cell carcinoma',
    'Unknown' # Used for unlabelled scans.
]

classes_abbrev = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]

# Final classes dictionary which excludes "Unknown" classes.
CLASSES_DICT = dict(tuple(zip(classes_abbrev[:-1], classes[:-1])))

seed = 42
n_splits = 1
batch_size = 16
epochs = 50
encoder_name = "timm-efficientnet-b7"
in_channels = 3
depth = 5
pretrained_weights = "noisy-student"
in_features = 1024
num_workers = 4

training_id = "training-{}-decoder-{}-channels-{}-batch_size".format(encoder_name, in_channels, batch_size)
log_name = "{}.log".format(training_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_lr = 1e-4 # 0.0001

lr_scheduler = "cosine"

loss = 'categorical'

# --------

train_df_path = "train_df.csv"
val_df_path = "val_df.csv"

train_df = pd.read_csv(train_df_path) # Labels and filenames
validation_df = pd.read_csv(val_df_path) # Metadata DF


print("-----------------Training utils--------------------")
print("-----------------\nTraining DataFrame: \n{}".format(train_df.head()))
print("-----------------\nValidation DataFrame: \n{}".format(validation_df.head()))
print("Encoder Name: \n{}".format(encoder_name))
print("LOG NAME: \n{}".format(log_name))
print("`init` (start) learning rate: {}".format(init_lr))

"""
def write_log(log_name=log_name, log):
	with open(log_name, 'w') as f:
		f.write(log)


if os.path.isfile(log_name):
	print("Log is available, not creating new one: {}".format(log_name))

else:
	print("Creating the log file: {}".format(log_name))
	with open(log_name, 'x') as log:
		log.write("This is the training log for training id: {}".format(training_id))
	print("log created.")

"""

print("Training starting...")

training_dataset = SkinCancerDatasetRetriever(
					df=train_df,
					images_dir="",
					image_size=512,
					mode = 'train')

validation_dataset = SkinCancerDatasetRetriever(
					df=validation_df,
					images_dir="",
					image_size=512,
					mode = 'train')

train_loader = DataLoader(training_dataset, 
						  batch_size=batch_size,
						  sampler=RandomSampler(training_dataset), 
						  num_workers=num_workers, 
						  drop_last=True)

valid_loader = DataLoader(validation_dataset, 
						  batch_size=batch_size, 
						  sampler=SequentialSampler(validation_dataset), 
						  num_workers=num_workers, 
						  drop_last=False)

print('TRAIN len(): {} | VALID len(): {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))


'''model = EfficientNetB7ClsHead(
						in_features=in_features,
						out_features=8,
						in_channels = 3,
						encoder_name="timm-efficientnet-b7",
						pretrained_weights='noisy-student',
						depth = 5
						)'''




#- --------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init
import os
import cv2
import timm
import torch
import random
import sklearn
import numpy as np
import pandas as pd
import albumentations as A
import segmentation_models_pytorch


# Specific Imports.
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit
from segmentation_models_pytorch.encoders import get_encoder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from segmentation_models_pytorch.base import initialization as init
from torch.utils.data.sampler import SequentialSampler, RandomSampler
efnb7_noisy_student_encoder = get_encoder(encoder_name, 
                                          in_channels=in_channels,
                                          depth=depth,
                                          weights=pretrained_weights)

class EfficientNetB7ClsHead(nn.Module):
    def __init__(self, encoder, in_features):
        super(EfficientNetB7ClsHead, self).__init__()
        self.encoder = encoder
        self.flatten_block = nn.Sequential(*list(self.encoder.children())[-4:])
        
        # Note: There seems to be a problem when I just slice the list of children layers. 
        # Deletion works however.
        del self.encoder.global_pool
        del self.encoder.act2
        del self.encoder.bn2
        del self.encoder.conv_head
        
        self.fc = nn.Linear(2560, in_features, bias=True)  
        self.cls_head = nn.Linear(in_features, len(CLASSES_DICT.keys()), bias=True)
        
        # Xavier uniform weight initialization.
        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
    
    @autocast
    def forward(self, x):
        x = self.encoder(x)[-1]  # Output shape: (batch_size, 640, 16, 16).
        x = self.flatten_block(x)  # Output shape: (batch_size, 2560).
        x = self.fc(x)  # Output shape: (batch_size, 1024).
        x = F.relu(x)  # Output shape: (batch_size, 1024).
        x = F.dropout(x, p=0.5, training=self.training)  # Output shape: (batch_size, 1024).
        x = self.cls_head(x)  # Output shape: (batch_size, 8).
        return x
    
model = EfficientNetB7ClsHead(efnb7_noisy_student_encoder, in_features=in_features)

#- --------------------



model.to(device)
CHECKPOINT = '{}_{}_{}_SKIN_CANCER.pth'.format(encoder_name, init_lr , lr_scheduler)


if "categorical" in loss:
	main_criterion = nn.CrossEntropyLoss()
# might wanna use different losses


optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)

scaler = torch.cuda.amp.GradScaler()



if "cosine" in lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)


log_file = open(log_name, "a")
log_file.write('epoch, lr, train_loss, val loss\n')
log_file.close()

best_epoch = 0
count = 0

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
	print("Epoch: {}".format(epoch))

	loss = 0.0
	for images, labels in train_loader:
		images.to(device)
		labels.to(device)

		optimizer.zero_grad()

		outputs = model(images)

		loss = criterion(outputs, labels)

		loss.backward()
		optimizer.step()

		print("Loss:","\n",loss)






