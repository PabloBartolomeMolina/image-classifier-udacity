import argparse
import torch
from torch import nn, tensor, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import time

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./vgg16_checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10) # 3 for testing changes, 10 as final to get some extra accuracy
    parser.add_argument('--gpu', dest="gpu", action="store", default="cuda")
    args = parser.parse_args()
    return args
	
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def valid_transformer(test_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(test_dir, transform=valid_transforms)
    return valid_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def data_loader(data, train=True):
    if train:
        b_size = 64
        loader = torch.utils.data.DataLoader(data, batch_size=b_size, shuffle=True)
    else:
        b_size = 64
        loader = torch.utils.data.DataLoader(data, batch_size=b_size)    
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    
    return device

def primaryloader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    for param in model.parameters():
        param.requires_grad = False 
    return model
    
def initial_classifier(model, hidden_units):
    from collections import OrderedDict

    classifier_layers = OrderedDict({
        'inputs' : nn.Linear(25088, 4096),
        'relu1': nn.ReLU(),
        'dropout': nn.Dropout(0.05),
        'output' : nn.Linear(4096, 102),
        'softmax': nn.LogSoftmax(dim=1)
    })
    classifier = nn.Sequential(classifier_layers)

    model.classifier = classifier
    return classifier
    
def network_trainer_valider(model, train_loader, validation_loader, device, 
                  criterion, optimizer, epochs, print_every):
    
    if type(epochs) == type(None):
        epochs = 10
        print(f'Number of epochs specificed as {epochs}:.0f.')    
 
    print("Training process initializing .....\n")
    
    running_loss = running_accuracy = 0
    validation_losses, training_losses = [],[]
    start = time.time() # Time used as reference to see the time cost of training the model.

    # Train Model
    for e in range(epochs):
        batches = 0 # 1 batch = 64 images

        model.train() # Training phase.

        for images,labels in train_loader:
            # Print out partial metrics and time to see progress and bottlenecks.
            if (batches == 0) and (e == 0):
                print(f'Epoch {e+1}/{epochs} | Batch {batches} | Time diff from previous batch {0.0:.4f} seconds')
            else:
                print(f'Epoch {e+1}/{epochs} | Batch {batches} | Time diff from previous batch {time.time() - start:.4f} seconds')
            start = time.time() # defines start time
            batches += 1

            images,labels = images.to(device),labels.to(device)

            # Pushes batch through network forward and then get the gradient backwards to update the weights.
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()

            # Calculates the metrics.
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1,dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # Resets optimiser gradient and tracks metrics.
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            # runs the model on the validation set every 5 loops
            if batches%print_every == 0:
                end = time.time()
                training_time = end-start
                start = time.time()

                # Resets the metrics
                validation_loss = 0
                validation_accuracy = 0

                model.eval()  # Evaluation phase.
                with torch.no_grad(): # Save memory - turns off calculation of gradients.
                    for images,labels in validation_loader:
                        images,labels = images.to(device),labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps,labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1,dim=1)
                        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()

                        # Update validation metrics (used to see progress of the model every 20 steps).
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()
                    
                # Update training metrics.
                end = time.time()
                validation_time = end-start
                validation_losses.append(running_loss/print_every)
                training_losses.append(validation_loss/len(validation_loader))
                    
                # Prints out metrics to track the progress.
                print(f'Epoch {e+1}/{epochs} | Batch {batches}')
                print(f'Time {validation_time}')
                print(f'Running Training Loss: {running_loss/print_every:.3f}')
                print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(f'Validation Loss: {validation_loss/len(validation_loader):.3f}')
                print(f'Validation Accuracy: {validation_accuracy/len(validation_loader)*100:.2f}%')

                # Resets the metrics.
                running_loss = running_accuracy = 0
                model.train()  # Training phase.

    return model

# saves the model's state_dict
def save_model(trained_model,destination_directory,model_arch):
    # Defines model's checkpoint.
    # - General improvement :
    #       fixed values of neurons -> variables for each at the beginning of the file to easily manage changes.
    print(trained_model.class_to_idx)
    model_checkpoint = {'model_arch':model_arch, 
                    'clf_input':25088,
                    'clf_output':102,
                    'clf_hidden':4096,
                    'state_dict':trained_model.state_dict(),
                    'model_class_to_index':trained_model.class_to_idx,
                    #'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': 0.003,
                    'train_batch_size': 64,
                    'val_batch_size': 32,
                    'test_batch_size': 32,
                    'epochs_trained': 20,
                    }
    checkpoint = {'architecture': trained_model.name,
                  'classifier': trained_model.classifier,
                  'class_to_idx': trained_model.class_to_idx,
                  'state_dict': trained_model.state_dict()}
                          
    if destination_directory: # If defined, we save in a specific directory.
        #torch.save(model_checkpoint,destination_directory+"/"+model_arch+"_checkpoint.pth")
        torch.save(checkpoint,destination_directory+"/"+model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to {destination_directory}")
    else: # current directory
        torch.save(checkpoint,model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to current directory as {model_arch}_checkpoint.pth")

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Verify if GPU was requested and availability.
    device = check_gpu(gpu_arg=args.gpu);
    
    # Recover data through trainloader applying transforms. First define the datapaths.
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Set up model and pass it to GPU / CPU.
    model = primaryloader_model(architecture=args.arch)
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    model.to(device);
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.003   # Default value choosen for this application.
        print(f'Learning rate specificed as {learning_rate:.3f}')
    else:
        learning_rate = args.learning_rate
        print(f'Learning rate specificed as {learning_rate:.3f}')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train the model with all needed parameters set.
    print_every = 20
    trained_model = network_trainer_valider(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every)
    print("\nTraining process and validation are completed!!")
    
    # Call to save the model.
    # Use current directory by default.
    destination_directory = None
    trained_model.class_to_idx = train_data.class_to_idx # improves label to name mapping
    print(trained_model.class_to_idx)
    save_model(trained_model,destination_directory,'vgg16')
if __name__ == '__main__': main()