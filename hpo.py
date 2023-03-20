#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ====================================#
# 1. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch as smd

def test(model, test_loader, criterion, hpo, hook=None):
    print("Testing Model on Testing Dataset")
    model.eval()
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    running_samples=0
    
    for inputs, labels in test_loader:
        #inputs=inputs.to(device)
        #labels=labels.to(device)
        
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        
        running_samples+=len(inputs)
        
        # Early stopping so that we don't test the whole dataset, especially for hyperparameter tuning
        if hpo == 1:
            if running_samples>(0.005*len(test_loader.dataset)):
                break

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")

def train(model, train_loader, validation_loader, criterion, optimizer, hpo, hook=None):
    epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(1, epochs+1):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            # =================================================#
            # 2. Set the SMDebug hook for the training phase. #
            # =================================================#
            if phase=='train':
                model.train()
                grad_enabled = True
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                grad_enabled = False
                hook.set_mode(smd.modes.TRAIN)
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                #inputs=inputs.to(device)
                #labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                # Check project 4 for reference
                if hpo == 1:
                    if running_samples>(0.2*len(image_dataset[phase].dataset)):
                        break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            
            status = '{} loss: {:.4f}, acc: {:.4f}, best validation loss: {:.4f}'.format(phase,epoch_loss,epoch_acc,best_loss)
            logger.info(status)

        if loss_counter==1:
            break
    return model
    
def net():
    '''
    Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # weights doesn't work in environment deployed through training job - seems to be 
    # an old version.
    #model = models.resnet50(weights='IMAGENET1K_V2')
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   #nn.Linear(num_features, 133))
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    

    
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
    
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,shuffle=True)
    
    return train_data_loader,test_data_loader,validation_data_loader

# Could be needed, from live class
# instead of inference script, like in examples
def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

#def save_model(model, model_dir, model_name):
#    logger.info("Saving the model.")
#    path = os.path.join(model_dir, model_name)
#    torch.save(model.cpu().state_dict(), path)
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    
    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
                          model.fc.parameters(), 
                          lr=args.lr
                          )
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_data_loader, test_data_loader, validation_data_loader=create_data_loaders(args.data, args.batch_size)
    
    # ===========================================================#
    # 5. Pass the SMDebug hook to the train and test functions. #
    # ===========================================================#
    model=train(model, train_data_loader, validation_data_loader, loss_criterion, optimizer, args.hpo, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_data_loader, loss_criterion, args.hpo, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    #save_model(model, args.model_dir, "model.pth")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default 1.0)"
    )
    
    parser.add_argument('--hpo', type=int, default=1) 
    
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING']) 
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    
    logger.info("Setup configuration: {}".format(args))
    logger.info(f"HPO mode: {args.hpo}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    
    main(args)
