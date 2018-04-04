"""Params for ADDA."""

# params for dataset and data loader
data_root = "/media/Data/dataset_xian/VisDA/"
dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
batch_size = 128
image_size = 224
num_workers = 2
gpu_ids = [0,1]

# params for source dataset
# src_encoder_restore = '/home/zxf/.torch/models/resnet34-333f7ec4.pth'
src_encoder_restore = 'snapshots/0404_1e3/CORAL-source-encoder-8.pt'
src_classifier_restore = 'snapshots/0404_1e3/CORAL-source-classifier-8.pt'
src_model_trained = True
CORAL_weight = 1e2

# params for training network
num_epochs_pre = 10
log_step_pre = 20
eval_epoch_pre = 2
save_epoch_pre = 2

model_root = 'snapshots/0404_1e2'
manual_seed = None

# params for optimizing models
learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
