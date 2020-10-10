import cv2
#from PIL import Image
import math
import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.optim as optim

from tool.darknet2pytorch import Darknet
from utils.utils_coco import *
from utils.utils import *
from models import *

import sys
import mmcv
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
sys.path.append('./mmdetection/')
from mmdet.datasets.pipelines import Compose
from mmdet import __version__
from mmdet.apis.inference import init_detector, LoadImage
#torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=1, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--max_iter', type=int, default=250, help='max number of iterations to find adversarial example')
parser.add_argument('--conf_thresh', type=float, default=0.5, help='conf thresh')
parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS thresh')
parser.add_argument('--im_size', type=int, default=500, help='the height / width of the input image to network')
parser.add_argument('--max_p', type=int, default=4700, help='max number of pixels that can change')
parser.add_argument('--minN', type=int, default=0, help='min idx of images')
parser.add_argument('--maxN', type=int, default=999, help='max idx of images')
parser.add_argument('--save', default='select1000_new_p3', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.save, exist_ok=True)

pre = transforms.Compose([transforms.ToTensor()])
nor = transforms.Normalize([123.675/255., 116.28/255., 103.53/255.],[58.395/255., 57.12/255., 57.375/255.])

model1 = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
pretrained_dict = torch.load('checkpoints/yolov4.pth', map_location=torch.device('cuda'))
model1.load_state_dict(pretrained_dict)
model1.eval().cuda()

config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
meta = [{'filename': '../images/6.png', 'ori_filename': '../images/6.png', 'ori_shape': (500, 500, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]
model2 = init_detector(config, checkpoint, device='cuda:0')

cfg = model2.cfg
device = next(model2.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
#print(test_pipeline)

def get_mask(image, meta, pixels):
    mask = torch.zeros((1,3,500,500)).cuda()
    bbox, label = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
    bbox = bbox[bbox[:,4]>0.3]
    num = bbox.shape[0]
    if num > 10: num = 10
    if num == 0: return mask.float().cuda()
    lp = int(pixels / (3*num))
    for i in range(num):
        xc = int((bbox[i,0]+bbox[i,2])/2)
        yc = int((bbox[i,1]+bbox[i,3])/2)
        w = int(bbox[i,2]-bbox[i,0])
        h = int(bbox[i,3]-bbox[i,1])
        lw = int(w/(w+h)*lp)
        lh = int(h/(w+h)*lp)
        y1 = max(0, yc-lw//4)
        y2 = min(yc+lw//4, 500)
        x1 = max(0, xc-lh//4)
        x2 = min(xc+lh//4, 500)
        mask[:,:,y1:y2,xc-1:xc+2] = 1
        mask[:,:,y1:y1+3,x1:xc+1] = 1
        mask[:,:,y2-2:y2+1,xc-1:x2] = 1
        mask[:,:,yc-1:yc+2,x1:x2+1] = 1
        mask[:,:,yc-1:yc+lh//4,x1:x1+3] = 1
        mask[:,:,yc-lh//4:yc+2,x2-2:x2+1] = 1
        
    mask = mask.float().cuda()    

    return mask

def get_mask2(image, meta, pixels):
    mask = torch.zeros((1,3,500,500)).cuda()
    bbox, label = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
    bbox = bbox[bbox[:,4]>0.3]
    num = bbox.shape[0]
    if num > 10: num = 10
    if num == 0: return mask.float().cuda()
    lp = int(pixels / (3*num))
    for i in range(num):
        xc = int((bbox[i,0]+bbox[i,2])/2)
        yc = int((bbox[i,1]+bbox[i,3])/2)
        w = int(bbox[i,2]-bbox[i,0])
        h = int(bbox[i,3]-bbox[i,1])
        lh = int(w/(w+h)*lp)
        lw = int(h/(w+h)*lp)
        y1 = max(0, yc-lw//4)
        y2 = min(yc+lw//4, 500)
        x1 = max(0, xc-lh//4)
        x2 = min(xc+lh//4, 500)
        mask[:,:,y1:y2,xc-1:xc+2] = 1
        mask[:,:,y1:y1+3,x1:xc+1] = 1
        mask[:,:,y2-2:y2+1,xc-1:x2] = 1
        mask[:,:,yc-1:yc+2,x1:x2+1] = 1
        mask[:,:,yc-1:yc+lh//4,x1:x1+3] = 1
        mask[:,:,yc-lh//4:yc+2,x2-2:x2+1] = 1        
        
    mask = mask.float().cuda()    

    return mask


files = os.listdir('../images')
files.sort()
#files = files[100:101]
#files = ['6.png']   
count = 0
count2 = 0
shape1 = 0
shape2 = 0
num1 = 0
num2 = 0
num3 = 0
pixels = 1800
for file in files:
    flag = 0
    if file == '4361.png':
        count += 1
        continue
    if count < args.minN:
        count += 1
        continue
    if count > args.maxN:
        break
    print(file)
    # prepare data
    data = dict(img='../images/'+file)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]
    #data['img'] = data['img'][0]
    #data['img_metas'] = data['img_metas'][0]    

    mask1 = get_mask(data['img'], data['img_metas'], pixels)
    mask2 = get_mask2(data['img'], data['img_metas'], pixels)
    
    #mask1_save = mask1.clone().detach()
    #vutils.save_image(mask1_save, 'mask1.png')
    #mask2_save = mask2.clone().detach()
    #vutils.save_image(mask2_save, 'mask2.png')    
    
    img_pil = cv2.imread('../images/'+file)
    img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
    img_pil = np.transpose(img_pil, (2,0,1))
    img = torch.from_numpy(img_pil/255.).float()
    img = img.unsqueeze(0).cuda()   
    
    patch = img.clone().detach()
    patch.requires_grad = True
    optimizer = optim.SGD([patch], lr=32/255.)
    #optimizer = optim.SGD([patch], lr=256/255.)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [10, 25, 60, 100, 190], gamma = 0.5, last_epoch=-1)        
    for i in range(args.max_iter):
        scheduler.step()
        imgp = torch.mul(img, 1-mask1) + torch.mul(patch, mask1)
        Img1 = F.interpolate(imgp, size=(608, 608), mode='bilinear', align_corners=True)
        out1 = model1(Img1)
        
        Img2 = F.interpolate(imgp, size=(800, 800), mode='bilinear', align_corners=True)
        ImgNor = nor(Img2.squeeze(0))
        data['img'] = [ImgNor.unsqueeze(0)]
        out2 = model2(return_loss=False, rescale=True, img=data['img'], img_metas=data['img_metas'])
        #print(len(prop))
        #print(prop[0].shape)
        loss1 = torch.max(out1[1])
        loss2 = torch.max(out2[0][:,4])
        
        loss = loss1 + loss2
        print('Num:{:4d}, Iter:{:4d}, Loss:{:.4f}, LossYOLO:{:.4f}, LossFRCNN:{:.4f}'.format(count, i, loss.item(), loss1.item(), loss2.item()))
        if loss1.item() < 0.45 and loss2.item() < 0.26:
            flag = 1
            num1 += 1
            shape1 += 1
            break
        
        optimizer.zero_grad()
        loss.backward()
        patch.grad = torch.sign(patch.grad)
        optimizer.step()
        patch.data.clamp_(0, 1)
    record1 = loss2.item()
    
    if flag == 0:
        patch = img.clone().detach()
        patch.requires_grad = True
        optimizer = optim.SGD([patch], lr=32/255.)
        #optimizer = optim.SGD([patch], lr=256/255.)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [10, 25, 60, 100, 190], gamma = 0.5, last_epoch=-1)        
        for i in range(args.max_iter):
            scheduler.step()
            imgp = torch.mul(img, 1-mask2) + torch.mul(patch, mask2)
            Img1 = F.interpolate(imgp, size=(608, 608), mode='bilinear', align_corners=True)
            out1 = model1(Img1)
            
            Img2 = F.interpolate(imgp, size=(800, 800), mode='bilinear', align_corners=True)
            ImgNor = nor(Img2.squeeze(0))
            data['img'] = [ImgNor.unsqueeze(0)]
            out2 = model2(return_loss=False, rescale=True, img=data['img'], img_metas=data['img_metas'])
                    
            loss1 = torch.max(out1[1])
            loss2 = torch.max(out2[0][:,4])
            
            loss = loss1 + loss2
            print('REmask==Num:{:4d}, Iter:{:4d}, Loss:{:.4f}, LossYOLO:{:.4f}, LossFRCNN:{:.4f}'.format(count, i, loss.item(), loss1.item(), loss2.item()))
            if loss1.item() < 0.45 and loss2.item() < 0.26:
                flag = 1
                num1 += 1
                shape2 += 1
                break
            
            optimizer.zero_grad()
            loss.backward()
            patch.grad = torch.sign(patch.grad)
            optimizer.step()
            patch.data.clamp_(0, 1)        
        record2 = loss2.item()

    if flag == 0:
        count2 += 1
        patch = img.clone().detach()
        patch.requires_grad = True        
        optimizer = optim.SGD([patch], lr=32/255.)
        #optimizer = optim.SGD([patch], lr=256/255.)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [10, 25, 50, 85, 125], gamma = 0.5, last_epoch=-1)
        if record1 > record2:
            mask = mask2.clone().detach()
        else:
            mask = mask1.clone().detach()
        for i in range(150):
            scheduler.step()
            imgp = torch.mul(img, 1-mask) + torch.mul(patch, mask)
            Img1 = F.interpolate(imgp, size=(608, 608), mode='bilinear', align_corners=True)
            out1 = model1(Img1)
            
            Img2 = F.interpolate(imgp, size=(800, 800), mode='bilinear', align_corners=True)
            ImgNor = nor(Img2.squeeze(0))
            data['img'] = [ImgNor.unsqueeze(0)]
            out2 = model2(return_loss=False, rescale=True, img=data['img'], img_metas=data['img_metas'])
                    
            loss1 = torch.max(out1[1])
            loss4 = torch.mean(out2[0][:,4])
            loss3 = torch.sum(out2[0][:,4]>0.3)
            loss2 = torch.max(out2[0][:,4])
            
            loss = loss1 + loss4 + loss3
            print('Num:{:4d}, Iter:{:4d}, Loss:{:.4f}, LossYOLO:{:.4f}, LossFRCNN:{:.4f}, BoxNum:{:.2f}'.format(count, i, loss.item(), loss1.item(), loss4.item(), loss3.item()))
            if loss1.item() < 0.45 and loss2.item() < 0.26:
                flag = 1
                break
                
            optimizer.zero_grad()
            loss.backward()
            patch.grad = torch.sign(patch.grad)        
            optimizer.step()
            patch.data.clamp_(0, 1)        
        
    if loss1.item() < 0.45: num2 += 1
    if loss2.item() < 0.3: num3 += 1
    
    count += 1
    print('-'*25,num1)
    
    imgp_save = imgp.clone().detach()
    vutils.save_image(imgp_save, args.save+'/'+file)
    
print(num1, num2, num3)
print(count2)
print(shape1, shape2)
torch.cuda.empty_cache()    

