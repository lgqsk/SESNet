import torch
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics, get_index
import torch.autograd as autograd
import logging
from sklearn.metrics import confusion_matrix as conmatrix

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

model = torch.load("network.pt")    # Loading Network
test_loader = get_test_loaders(opt) # Loading datasets
test_metrics = initialize_metrics() # Initializing the confusion matrix

model.eval()
with torch.no_grad():
    for batch_img1, batch_img2, labels in test_loader:

        # Set variables for testing
        batch_img1 = autograd.Variable(batch_img1).float().to(dev)
        batch_img2 = autograd.Variable(batch_img2).float().to(dev)
        labels = autograd.Variable(labels).long().to(dev)

        # Get prediction map _m
        cd_preds,_ = model(batch_img1, batch_img2)
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate the confusion matrix
        cd_val_report = conmatrix(labels.data.cpu().numpy().flatten(),
                                    cd_preds.data.cpu().numpy().flatten(),
                                    labels=[1,0])

        # Confusion matrix accumulation
        test_metrics += cd_val_report
    
    # Generate evaluation metrics
    test_index = get_index(test_metrics)

    # Displays F1_score, Precision, Recall, IoU and OA in order
    print("Displays F1_score, Precision, Recall, IoU and OA in order: ")
    logging.info("EPOCH {} TEST METRICS".format(test_index))