import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


encoder_name = "timm-efficientnet-b7"
in_channels = 3
depth = 5
pretrained_weights = "noisy-student"
in_features = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_lr = 0.0001


class EfficientNetB7ClsHead(nn.Module):
    def __init__(
                 self,
                 #encoder,
                 in_features,
                 out_features=8,
                 in_channels = 3,
                 encoder_name="timm-efficientnet-b7",
                 pretrained_weights='noisy-student',
                 depth = 5
                 ):


        super(EfficientNetB7ClsHead, self).__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.depth = depth
        self.pretrained_weights = pretrained_weights

        #----- Encoder ------
        self.encoder = get_encoder(self.encoder_name,
                              in_channels=self.in_channels,
                              depth=self.depth,
                              weights=self.pretrained_weights)

        #--------------------
        #self.encoder = encoder

        self.flatten_block = nn.Sequential(*list(self.encoder.children())[-4:])
        
        # Note: There seems to be a problem when I just slice the list of children layers. 
        # Deletion works however.
        del self.encoder.global_pool
        del self.encoder.act2
        del self.encoder.bn2
        del self.encoder.conv_head
        
        self.fc = nn.Linear(2560, in_features, bias=True)  
        self.cls_head = nn.Linear(in_features, out_features, bias=True)
        
        # Xavier uniform weight initialization.
        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
    
    
    #-----------------------

    #@torch.cuda.amp.autocast
    def forward(self, x):
        x = self.encoder(x)[-1]  # Output shape: (batch_size, 640, 16, 16).
        x = self.flatten_block(x)  # Output shape: (batch_size, 2560).
        x = self.fc(x)  # Output shape: (batch_size, 1024).
        x = F.relu(x)  # Output shape: (batch_size, 1024).
        x = F.dropout(x, p=0.5, training=self.training)  # Output shape: (batch_size, 1024).
        x = self.cls_head(x)  # Output shape: (batch_size, 8).
        return x
 

if __name__ == '__main__':
	model = EfficientNetB7ClsHead(
				in_features=1024,
				out_features=8,
				in_channels = 3,
				encoder_name="timm-efficientnet-b7",
				pretrained_weights='noisy-student',
				depth = 5)
	print(model)
	BS, C, H, W = 1, 3, 512, 512
	img = torch.randn(BS, C, H, W)
	pred = model(img)

	print(pred)



