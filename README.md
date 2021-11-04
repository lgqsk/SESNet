# SESNet for remote sensing image change detection

It is the implementation of the paper: "SESNet: A Semantically Enhanced Siamese Network for Remote Sensing Change Detection". Here, we provide the pytorch implementation of this paper.


## Prerequisites

- windows or Linux 
- PyTorch-1.4.0
- Python 3.6
- CPU or NVIDIA GPU

## Training

You can run a demo to start training.

```
python train.py
```

The network with the highest F1 score in the validation set will be saved in the folder `tmp`.

## testing

You can run a demo to start testing.
```
python test.py
```

The `F1_score`, `precision`, `recall`, `IoU` and `OA` are displayed in order.
Of course, you can slightly modify the code in the `test.py` file to save the confusion matrix.

## Prepare Datasets

### download the change detection dataset

SVCD is from the paper `CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS`, You could download the dataset at https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9;

LEVIR-CD is from the paper `A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection`, You could download the dataset at https://justchenhao.github.io/LEVIR/;

Take SVCD as an example, the path list in the downloaded folder is as follows:

```
├SVCD:
├  ├─train
├  │  ├─A
├  │  ├─B
├  │  ├─OUT
├  ├─val
├  │  ├─A
├  │  ├─B
├  │  ├─OUT
├  ├─test
├  │  ├─A
├  │  ├─B
├  │  ├─OUT
```

where A contains images of pre-phase, B contains images of post-phase, and OUT contains label maps.

When using the LEVIR-CD dataset, simply change the folder name from `SVCD` to `LEVIR`. The location of the dataset can be set in `dataset_dir` in the file `metadata.json`.

### cut bitemporal image pairs (LEVIR-CD)

The original image in LEVIR-CD has a size of 1024 * 1024, which will consume too much memory when training. In our paper, we cut the original image into patches of 256 * 256 size without overlapping.

When running our code, please make sure that the file path of the cut image matches ours.

## Define hyperparameters

The hyperparameters and dataset paths can be set in the file `metadata.json`.

```

"augmentation":  Data Enhancements
"num_gpus":      Number of simultaneous GPUs
"num_workers":   Number of simultaneous processes

"image_chanels": Number of channels of the image (3 for RGB images)
"init_channels": Adjust the overall number of channels in the network, the default is 32
"epochs":        Number of rounds of training
"batch_size":    Number of pictures in the same batch
"learning_rate": Learning Rate
"loss_function": The loss function is specified in the file `./utils/helpers.py`
"bilinear":      Up-sampling method of decoder feature maps, `False` means deconvolution, `True` means bilinear up-sampling

"dataset_dir":   Dataset path, "../SVCD/" means that the dataset `SVCD` is in the same directory as the folder `SESNet`.

```