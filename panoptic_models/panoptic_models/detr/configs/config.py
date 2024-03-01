# get absolute path to checkpoint
import os

# get directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
frozen_weights = dir_path + "/../weights/detr_detr_50_mask_mixed_1_epoch_250.pth"
lr_backbone = 0
backbone = "resnet50"
dilation = False
position_embedding = "sine"
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 100
pre_norm = False
masks = True
threshold = 0.85
aux_loss = True
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
mask_loss_coef = 1
dice_loss_coef = 1
bbox_loss_coef = 5
giou_loss_coef = 2
eos_coef = 0.1
dataset_file = "construction_site"
device = "cuda"
