import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models


from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import PIL
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.', default=1)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store", default="cuda")
    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path, device):
    # loads the GPU when available
    if device=="gpu":
        map_location=lambda device, loc: device.cuda()
    else:
        map_location='cpu'
        
    checkpoint = torch.load(f=checkpoint_path,map_location=map_location)
    # Get number of input units, output units, hidden units, and state_dict
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #print(model.class_to_idx)
    
    return model

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    processed_image = Image.open(image).convert('RGB') # open the image
    print(type(processed_image))
    resized_image = processed_image.resize((256, 256))
    width, height = resized_image.size   # Get dimensions of the resized image.
    
    # Crop the image's center.
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    cropped_image = resized_image.crop((left, top, right, bottom))
    # Debugging & informative purposes.
    #print(f"Width: {width}, Height: {height}")
    #print(f"Crop Box: Left={left}, Top={top}, Right={right}, Bottom={bottom}")
    
    # Converts to tensor and apply normalization.
    transf_tens = transforms.ToTensor()
    transf_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) # mean, std
    tensor = transf_norm(transf_tens(cropped_image))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean and std
    ])
    tensor_image = transform(cropped_image)
    
    # Convert tensor result to numpy array.
    np_processed_image = tensor_image.numpy()
    return np_processed_image

def class_to_label(file, classes):
    with open(file, 'r') as f:
        class_mapping =  json.load(f)
        
    labels = []
    for c in classes:
        labels.append(class_mapping[c])
    
    #print(class_mapping)
    #print(labels)
    return labels

# Predicts the class of an image using out deep learning model. 
def predict(image_path, model,idx_mapping, topk, device):
    # defines preprocessed image
    pre_processed_image = torch.from_numpy(process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image,0).to(device).float()
    
    model.to(device)
    model.eval()
    
    log_ps = model.forward(pre_processed_image)
    ps = torch.exp(log_ps)
    top_ps,top_idx = ps.topk(topk,dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    model.train()
    #print(list_idx)
    for x in list_idx:
        #print(x)
        classes.append(idx_mapping[x])
    return list_ps, classes

# Print out the probability output for the image selected and a given number of classes.
def print_predictions(probabilities, classes, image, category_names=None):
    print(image) # Image name is printed out, just informative.
    
    if category_names:
        labels = class_to_label(category_names,classes)
        for i,(ps,ls,cs) in enumerate(zip(probabilities,labels,classes),1):
            print(f'{i}) {ps*100:.2f}% {ls.title()} | Class No. {cs}')
    else:
        for i,(ps,cs) in enumerate(zip(probabilities,classes),1):
            print(f'{i}) {ps*100:.2f}% Class No. {cs} ')
    print('') 

def main():
    
    # Get arguments.
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
            
    print(f'image: {args.image}')
    # Select device and load model from checkpoint.
    device = check_gpu(gpu_arg=args.gpu);
    model = load_checkpoint(args.checkpoint, device)
    #print(model)
    
    idx_mapping = dict(map(reversed, model.class_to_idx.items()))
    #print(f'model.class_to_idx type: {type(model.class_to_idx.items())}, idx_mapping length: {len(model.class_to_idx.items())}')
    #print(f'idx_mapping type: {type(idx_mapping)}, idx_mapping length: {len(idx_mapping)}')
    
    # Processing of image and prediction.
    image_tensor = process_image(args.image)
    probabilities,classes = predict(args.image,model,idx_mapping,args.top_k,device)
    # Print results.
    print_predictions(probabilities,classes,args.image.split('/')[-1],'cat_to_name.json')

if __name__ == '__main__': main()