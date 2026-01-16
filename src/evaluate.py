import argparse
import pickle

import numpy as np
import torch

import json
import yaml
import os

from tqdm import tqdm

from experiment_configs import experiments_unsup_orig
from main_model import DGAD_Physio
from dataset import get_dataloader
from utils import train,  window_trick_evaluate_middle
from loaddataset import TrainData, get_mask, TestData
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description="DGAD")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--diffusion_step",type=int,default=50)
parser.add_argument("--machine_number",type=int,default=1)
parser.add_argument("--file",type=str)
#改改改
parser.add_argument('--dataset',type=str,default="amazon")
parser.add_argument('--sc',type=int,default=0)
parser.add_argument('--window_length',type=int,default=10)
parser.add_argument('--split',type=int,default=1)
parser.add_argument('--mask_ratio',type=float,default=0.5)
parser.add_argument('--mask_type',type=str,default="random")
parser.add_argument('--window_split',type=int,default="1")
args = parser.parse_args()


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)



# 由于是分开进行预测，
machine_number = args.machine_number

train_data_path_list = []
test_data_path_list = []
label_data_path_list = []


try:
    os.mkdir("window_result")
except:
    pass

# for iteration in os.listdir("train_result"):
iteration='save0'
try:
    os.mkdir(f"window_result/{iteration}")
except:
    pass
##改改改
# for subset_name in os.listdir(f"train_result/{iteration}/"):
subset_name='datasets_amazon_origin_0_49_unconditional:True_split:2_diffusion_step:50'
data_id = subset_name.split("_unconditional")[0]

if "unconditional:True" in subset_name:
    unconditional = True
else:
    unconditional = False

split = 4
diffusion_step = int(subset_name.split("diffusion_step:")[-1])

train_data_path_list = []
test_data_path_list = []
label_data_path_list = []
exp=experiments_unsup_orig[0]
root_dir = './datasets/'
ds_trn_id, ds_vld_id, ds_tst_id = exp['setting']['ds_train'], exp['setting']['ds_valid'], exp['setting'][
    'ds_test']
traj_step_features = exp['setting']['ds']['trj_step_feat']
# data_file = f"{data_id}_train.pkl"
# train_data_path_list.append("data/Machine/" + data_file)
# test_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test.pkl"))
# label_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test_label.pkl"))
# train_data = TrainData(root_dir, file_id=ds_trn_id, mode='train', sc=args.sc, submode='',
#                        traj_step_features=traj_step_features, window_length=args.window_length,
#                        split=args.split, mask_ratio=args.mask_ratio)
#
# temp_dict = train_data.__getitem__(0)
# # index_temp = np.random.randint(0, train_data.length - train_data.window_length)
# # temp_dict = train_data.__getitem__(index_temp)
# observed_mask = temp_dict['observed_mask']
# valid_data = TrainData(root_dir, file_id=ds_vld_id, mode='valid', sc=args.sc, submode='',
#                        traj_step_features=traj_step_features, window_length=args.window_length,
#                        split=args.split, mask_ratio=args.mask_ratio)
# mask_list = []
#
# # for i in tqdm(range(0, 100)):
# for i in tqdm(range(0, train_data.window_length)):
#     mask_list.append(get_mask(observed_mask, mask_ratio=args.mask_ratio))
##改改改
test_file_1 = pickle.load(
    open(f"datasets/data_pkl/dbcargo_windows_length5_checkpoint_tst48_1_origin_0.pkl", "rb")
)
test_file_2 = pickle.load(
    open(f"datasets/data_pkl/dbcargo_windows_length5_checkpoint_tst48_2_origin_0.pkl", "rb")
)
# test_data_strategy_1 = TestData(root_dir, file_id=ds_tst_id, mode='test', sc=args.sc, submode='',
#                                 traj_step_features=traj_step_features, window_length=args.window_length,
#                                 split=args.split, window_split=args.window_split,
#                                 strategy=0, mask_list=mask_list)
# test_data_strategy_2 = TestData(root_dir, file_id=ds_tst_id, mode='test', sc=args.sc, submode='',
#                                 traj_step_features=traj_step_features, window_length=args.window_length,
#                                 split=args.split, window_split=args.window_split,
#                                 strategy=1, mask_list=mask_list)
# train_loader = DataLoader(train_data, batch_size=24, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=24, shuffle=True)

train_error_loader_list = DataLoader(test_file_1, batch_size=24)
test_loader_list = DataLoader(test_file_2, batch_size=24)

# # epoch = file.split("-")[0]
# train_data_path = train_data_path_list[0]
# test_data_path = test_data_path_list[0]
# label_data_path = label_data_path_list[0]
# train_loader, valid_loader, train_error_loader_list, test_loader_list = get_dataloader(
#     train_data_path,
#     test_data_path,
#     label_data_path,
#     batch_size=24,
#     window_split=2,
#     split=split
# )



# dataset_name = exp['setting']['ds']['dataset_name']
if 'yn' in args.dataset:
    bs = 256
    lr = 1e-03
    wd = 0.
    drp = 0.
    seq_len = 72
    seq_len_dk = seq_len  # 72
    num_heads = 12
    feature_dim = 2 * seq_len
    # feature_dim = 32

if 'amazon' in args.dataset:
    bs = 256
    lr = 1e-03
    wd = 0.
    drp = 0.
    seq_len = 72
    seq_len_dk = seq_len  # 72
    num_heads = 12
    feature_dim = 2 * seq_len

if 'brightkite' in args.dataset:
    bs = 256
    lr = 1e-04
    wd = 0.
    drp = 0.
    seq_len = 500
    seq_len_dk = seq_len  # 500
    num_heads = 8
    feature_dim = 2 * seq_len

if 'dbcargo' in args.dataset:
    bs = 256
    # lr = (1e-02, 1e-03)['RU' in model_name]
    lr = (1e-02, 1e-03)
    wd = 0.
    drp = 0.
    seq_len = 72
    seq_len_dk = seq_len  # 72
    num_heads = 12
    feature_dim = 2 * seq_len
    # feature_dim = 32


config["model"]["is_unconditional"] = unconditional
model = DGAD_Physio(config, args.device, target_dim=feature_dim, ratio=args.ratio).to(args.device)
base_folder = f"train_result/{iteration}/{subset_name}"

model.load_state_dict(torch.load(f"{base_folder}/best-model.pth",map_location=args.device))

print("base folder is ")
print(base_folder)
if not os.path.exists(f"window_result/{iteration}/{diffusion_step}"):
    os.mkdir(f"window_result/{iteration}/{diffusion_step}")
# try:
#     os.mkdir(f"window_result/{iteration}/{diffusion_step}")
# except:
#     pass

target_folder = f"window_result/{iteration}/{diffusion_step}/{subset_name}"
if not os.path.exists(target_folder):
    os.mkdir(target_folder)
# try:
# #     os.mkdir(target_folder)
# except:
#     continue

for temp_i in range(0,1):
    window_trick_evaluate_middle(model, train_error_loader_list, test_loader_list, nsample=1, scaler=1,
                      foldername=target_folder,
                      epoch_number=0, name=str(temp_i),split=split)