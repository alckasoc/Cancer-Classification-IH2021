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
		super(SiimCovidAuxDataset, self).__init__()	
		self.df = df
		self.images_dir = images_dir
		self.image_size = image_size
		assert mode in ['train', 'valid']
		self.mode = mode

		if self.mode == 'train':
			self.transform = albu.Compose([
				albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
				albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),
				albu.HorizontalFlip(p=0.5),
				albu.VerticalFlip(p=0.5),
				albu.OneOf([
					albu.MotionBlur(p=.2),
					albu.MedianBlur(blur_limit=3, p=0.1),
					albu.Blur(blur_limit=3, p=0.1),
				], p=0.25),
				albu.OneOf([
					albu.CLAHE(clip_limit=2),
					albu.IAASharpen(),
					albu.IAAEmboss(),
					albu.RandomBrightnessContrast(),            
				], p=0.25),
				albu.Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
				albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
				ToTensorV2(),
			])
		
		# infer or validation data
		else:
			self.transform = albu.Compose([
				albu.Resize(self.image_size, self.image_size),
				albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
				ToTensorV2(),
			])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		#           /kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000000.jpg
		img_path = '/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{}.jpg'.format(self.df['Filename'].values[index])

		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (self.image_size, self.image_size)) #2, 3, 512, 512

		label = torch.FloatTensor(self.df.loc[index, classes])

		transformed = self.transform(image=image)
		image = transformed["image"]
		
		
		if self.mode == 'train':
			return image, label
		else:
			return image





