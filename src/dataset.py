import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


classes = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]

# Dataset retriever for `skin cancer` classification
class SkinCancerDatasetRetriever(torch.utils.data.Dataset):
	def __init__(self, df, images_dir, image_size, mode):
		super(SkinCancerDatasetRetriever, self).__init__()	
		self.df = df
		self.images_dir = images_dir
		self.image_size = image_size
		assert mode in ['train', 'valid']
		self.mode = mode

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		#           /kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000000.jpg
		img_path = '/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{}.jpg'.format(self.df['image'].values[index])

		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (self.image_size, self.image_size)) #2, 3, 512, 512

		label = torch.FloatTensor(self.df.loc[index, classes])

		#transformed = self.transform(image=image)
		#image = transformed["image"]
		
		
		if self.mode == 'train':
			return image, label
		else:
			return image





