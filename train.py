import datetime
import torch
import torch.utils.data
import numpy as np
import random
import torch.optim as optim
import torch.autograd as autograd
from sklearn.metrics import confusion_matrix as conmatrix
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,get_test_loaders,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics,get_index)
from utils.adamW import AdamW
import os
import logging
import json
from tensorboardX import SummaryWriter
from models.DeepSupervision import DSNet

# Initialize Parser and define arguments
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

# Initialize experiments log
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

# Set up environment: define paths, download data, and set device
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

# Fixing parameters with seeds
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)

# Load datasets
train_loader, val_loader = get_loaders(opt)

# Load Model
logging.info('LOADING Model')
model = load_model(opt, dev)
DSNet = DSNet(model).to(dev)

# Define loss function, optimizer, learning rate decay strategy
criterion = get_criterion(opt)
optimizer = AdamW(DSNet.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


# best_metrics is the test set indicator corresponding to the optimal epoch of the validation set
best_metrics = np.array([-1,-1,-1,-1])


logging.info('STARTING training')
for epoch in range(opt.epochs):
    
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    DSNet.train()        
    for batch_img1, batch_img2, labels in train_loader:
        
        # Set variables for training
        batch_img1 = autograd.Variable(batch_img1).float().to(dev)
        batch_img2 = autograd.Variable(batch_img2).float().to(dev)
        labels = autograd.Variable(labels).long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()
        
        # Get the loss value of the main network and deep supervision
        loss_main, loss_DS = DSNet(batch_img1, batch_img2, labels)
        
        # Get the overall loss value
        loss = loss_main + loss_DS

        # Back propagation
        loss.backward()
        optimizer.step()
        
        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            
            # Set variables for val
            batch_img1 = autograd.Variable(batch_img1).float().to(dev)
            batch_img2 = autograd.Variable(batch_img2).float().to(dev)
            labels = autograd.Variable(labels).long().to(dev)

            # Get predictions
            cd_preds,_ = model(batch_img1, batch_img2)
            _, cd_preds = torch.max(cd_preds, 1)
            cd_val_report = conmatrix(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 labels=[1,0])
    
            val_metrics += cd_val_report
        
        # acquire F1_score,precision,recall,OA
        index = get_index(val_metrics)
        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(index))
        
        """
        Store the weights of good epochs based on validation results
        """
        if index[0] > best_metrics[0]: # Based on F1-score

            best_metrics = index   

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            metadata['validation_metrics'] = index
            
            # Save
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/metadata.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, './tmp/network.pt')


            # for test dataset
            test_loader = get_test_loaders(opt)
            test_metrics = initialize_metrics()
            for batch_img1, batch_img2, labels in test_loader:
                # Set variables for testing
                batch_img1 = autograd.Variable(batch_img1).float().to(dev)
                batch_img2 = autograd.Variable(batch_img2).float().to(dev)
                labels = autograd.Variable(labels).long().to(dev)

                # Get predictions
                cd_preds,_ = model(batch_img1, batch_img2)
                _, cd_preds = torch.max(cd_preds, 1)
                cd_val_report = conmatrix(labels.data.cpu().numpy().flatten(),
                                            cd_preds.data.cpu().numpy().flatten(),
                                            labels=[1,0])
                test_metrics += cd_val_report

            test_index = get_index(test_metrics)
            logging.info("EPOCH {} TEST METRICS".format(test_index))

            # Insert training and epoch information to metadata dictionary
            metadata['test_metrics'] = test_index

            # Save test result
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/test_metadata.json', 'w') as fout:
                json.dump(metadata, fout)
            
# Test set results corresponding to the best model in the val set
logging.info("THE BEST TEST METRICS {}".format(test_index))
print("Displays F1_score, Precision, Recall, IoU and OA in order!")
writer.close()
print('Done!')