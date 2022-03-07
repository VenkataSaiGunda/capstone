# capstone
#Class project




import matplotlib.pyplot as plt
from PIL import Image
from random import randint

import pandas as pd
import numpy as np
from tqdm import tqdm

import urllib
import cv2


import torch
import torchvision

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# load train data
df_train = pd.read_csv('train.csv')
print(df_train.head())

# load boxes data and merge into one
df_boxes_split1 = pd.read_csv('boxes_split1.csv')
df_boxes_split2 = pd.read_csv('boxes_split2.csv')
df_boxes = pd.concat([df_boxes_split1, df_boxes_split2])

print(df_boxes.head())

df_train = pd.merge(df_train, df_boxes, on='id',  how='right')
df_train.head()



def get_transform(train):
    transforms = []
    if train:
        # random horizontal flip with 50% probability
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class GoogleLandmarks(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.dim = (512, 512)
        self.transforms = transforms
        self.ids = np.unique(df['landmark_id'].values)
        self.ids_dic = {v:k for k,v in enumerate(self.ids)}
    
    def url_to_image(self, url, dim):
        try:
            resp = urllib.request.urlopen(url)
        except:
            return np.array([])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        if(image.size != 0):
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            image = Image.fromarray(np.uint8(image))
            if(image):
                image = self.transforms(image)
        else:
            image = Image.fromarray(image)
        return T.ToTensor()(image)
    
    def get_rect(self, boxes):
        try:
            y = boxes[0]
            x = boxes[1]
            h = boxes[2] - boxes[0]
            w = boxes[3] - boxes[1]
        except:
            return None
        return plt.Rectangle((x, y), w, h, color='y', alpha=0.3)
    
    def draw_bbox(self, img, rect):
        fig, ax = plt.subplots()
        plt.imshow(img.permute(1, 2, 0))
        if(rect):
            ax.add_patch(rect)
    
    def format_boxes(self, boxes, dim):
        return (np.array(boxes.split(' ')).astype(np.float32) * dim[0]).astype(np.int64)
    
    def __getitem__(self, idx):
        id = self.df.iloc[idx].id
        landmarkid = self.df.iloc[idx].landmark_id
        url = self.df[self.df.id == id].url.values[0]
        boxes = self.df[self.df.id == id].box.values[0]
        
        
        # format boxes
        boxes = self.format_boxes(boxes, self.dim)
        
        labels = np.eye(len(self.ids))[self.ids_dic[landmarkid]]
        
        target = {}
        target["boxes"] = torch.as_tensor([boxes], dtype=torch.int64)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        
        image = self.url_to_image(url, self.dim)
        
        if(len(image) == 0):
            return None, None
        
        return image, target
        
    def __len__(self):
        return len(self.ids)













# select 10 ids randomly
idxes = [randint(0, len(df_train) - 1) for i in range(10)]

# select only 10 landmarks
ids_of_landmarks = df_train['landmark_id'][idxes].values

# subset of training data with 10 landmarks
df = df_train[df_train['landmark_id'].isin(ids_of_landmarks)]

# google dataset
google_ds = GoogleLandmarks(df, get_transform(train=True))
 



image, target = google_ds[0]







rect = google_ds.get_rect(target['boxes'][0])
google_ds.draw_bbox(image, rect)




def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = torch.utils.data.DataLoader(
        google_ds, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=collate_fn)
        
        
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(device)





params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)









total_errors = []
for epoch in range(10):
    losses_arr = []

    for images, targets in data_loader:

        images = list(image.to(device) for image in images if image is not None)
        targets = [{k: torch.as_tensor(v).detach().to(device) for k, v in t.items()} for t in targets if t is not None]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses_arr.append(losses.item())

        losses.backward()
        optimizer.step()

        # update the learning rate
        # lr_scheduler.step()
        
    total_errors.append(np.mean(np.array(losses_arr)))
    if epoch % 1 == 0:
        print("Epoch:{0:3d}, Loss:{1:1.3f}".format(epoch, total_errors[-1]))







plt.plot(total_errors)


