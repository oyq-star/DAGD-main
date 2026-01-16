import random

import torch
import numpy as np
import pandas as pd
# import umap
import umap.umap_ as umap

import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from random import sample
from experiment_configs import files
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertConfig



class TransformerEncoderModel(torch.nn.Module):
    def __init__(self, d_model=16, nhead=4, num_layers=2):
        super().__init__()
        self.embed = torch.nn.Linear(2, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embed(x.to(torch.float32))  # 形状：(batch_size, 72, d_model)
        x = self.transformer(x)  # 输出形状不变
        return x.mean(dim=1)
def get_traj_transformer_embeddings(traj_data_):
    encoder = TransformerEncoderModel(d_model=144)
    vector = encoder(traj_data_)
    return vector#
#采用预训练的上下文时间位置嵌入模型（Context-aware and Time-aware Location Embedding）
def get_traj_ctle_embeddings(coordinates):
    config = BertConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8
    )
    model = BertModel(config)

    # 添加时间编码特征
    timestamps = np.linspace(0, 1, 72)  # 标准化时间戳
    inputs = coordinates.flatten() + timestamps  # 时空联合编码
    outputs = model(inputs.unsqueeze(0))[1]  # 得到128维轨迹向量

def get_traj_embeddings(traj_data_, labels=None, verbose=False, seed=42, sc='robust'):
    
    traj_data_flat = traj_data_.reshape(-1, traj_data_.shape[1]*traj_data_.shape[2])
    
    if sc=='robust':
        scaler=RobustScaler()
    else:
        scaler=StandardScaler()    
    
    traj_data_scaled = scaler.fit_transform(traj_data_flat)
    
    # m = umap.UMAP(random_state=seed)
    m = umap.UMAP(n_components=32)
    traj_embeddings = m.fit_transform(traj_data_scaled)
    if verbose and labels is not None: plt.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1], c=labels)
    
    return traj_embeddings

def get_augmented_trajectories(traj_data_, aug_traj_len, num_traj_step_features, topknn_traj_indices, k):
        
    aug_trajectories=[]
    aug_traj_len = traj_data_.shape[1]
    aug_traj_seg_len = math.ceil(aug_traj_len / k)

    for idx_query_trj, trjknn in enumerate(topknn_traj_indices):
        
        # create augmented trajectory from segments of shuffled knn trajectories
        # todo: define seglen randomly and attach via round robin from knn trajectories until desired aug_tran_seg_len reached
        curstep,idx_knn=0,0
        aug_traj=torch.zeros((aug_traj_len, num_traj_step_features))
        while curstep < aug_traj_len:
            seglen = (aug_traj_seg_len,aug_traj_len-curstep)[(curstep+aug_traj_seg_len)>aug_traj_len]
            aug_traj[curstep:curstep+seglen] = torch.tensor(traj_data_[topknn_traj_indices[idx_query_trj, idx_knn], curstep:curstep+seglen,:])
            curstep=curstep+seglen
            idx_knn+=1
        aug_trajectories.append(aug_traj.numpy())
        
    return torch.tensor(aug_trajectories)

def load_trajectory_data(root_dir, file_id:int=0, mode:str='train', traj_step_features=['X_Coord','Y_Coord'], sc='robust',):
    
    filepath=f"{root_dir}{files[file_id]}"
    print(f'\n-------------Creating *{mode}* dataset from dataset file {filepath} -----\n')
    # 定义文件名列表
    if mode=='train' and 'brightkite' in files[file_id]:
        file_names = [
            f"{root_dir}0_{files[file_id]}",
            f"{root_dir}1_{files[file_id]}",
            f"{root_dir}2_{files[file_id]}",
            f"{root_dir}3_{files[file_id]}"
        ]

        # 初始化空列表，用于存储每个文件读取后的数据框
        data_frames = []

        # 遍历文件名列表
        for file_name in file_names:
            try:
                # 读取文件
                df = pd.read_csv(file_name)
                # 将读取的数据框添加到列表中
                data_frames.append(df)
            except FileNotFoundError:
                print(f"文件 {file_name} 未找到，请检查路径。")
            except Exception as e:
                print(f"读取文件 {file_name} 时发生错误: {e}")
        if data_frames:
            merged_df = pd.concat(data_frames, ignore_index=True)
            # 保存合并后的数据框为新的 CSV 文件
            # merged_df.to_csv('merged_brightkite.csv', index=False)
            print("文件合并完成，保存为 merged_brightkite.csv")
            df=merged_df
    df = pd.read_csv(filepath)
    labels = np.array(df.query("Step == 0")['Label']) # (1000,)
    
    print(df)
    
    file = filepath.split('/')[-1]

    if 'driver' in file:
        traj_len, total_trajectories = 72, int(file[-7:-4])
    
    elif 'bright' in file:
        traj_len, total_trajectories = int(file.split('_')[1]), int(file.split('_')[2])
        
    elif 'dbcargo' in file:
        traj_len, total_trajectories = int(file.split('_')[1]), int(file.split('_')[2])

    else: # synthetic, amazon
        _, traj_len, total_trajectories, _ = file.split('_')
        traj_len, total_trajectories = int(traj_len), int(total_trajectories)

    print(f"trajectory length: {traj_len}, abnormal trajectories: {np.sum(labels[:total_trajectories])}/{total_trajectories}")
    
    traj_Entity_IDs = np.array(df.query("Step == 0")['Entity'])
    
    coords=None # changed column names since original columns did not met policies
    for traj_entity_ID in traj_Entity_IDs: 

        df_temp=df.loc[df['Entity'] == traj_entity_ID] # trajectory steps of entity
        df_temp=df_temp.sort_values(by=['Step'])
        df_temp=df_temp[:traj_len] # restrict trajectory to max. traj_len steps
        df_temp=df_temp[traj_step_features].to_numpy()

        try:
            coords=np.vstack((coords,df_temp))
        except:
            coords=df_temp
#这段代码的目的是从数据框 df 中提取特定实体的轨迹步骤，对步骤进行排序，截断到指定长度，并提取指定的特征列，最后将结果转换为 NumPy 数组。
    # coords.shape (72000,2)

    if sc=='robust':
        scaler=RobustScaler() 
    else:
        scaler=StandardScaler() 

    scaler.fit(coords)
    coords_scaled = scaler.transform(coords)
    traj_data = np.reshape(coords_scaled, (len(traj_Entity_IDs), traj_len, len(traj_step_features)))
    # traj_data.shape (1000, 72, 2)

    traj_labels = labels # labels.shape (1000,)

    return traj_data, traj_labels



class TrainData(Dataset):

    def __init__(self,root_dir, file_id, mode, sc, submode='', traj_step_features=['X_Coord','Y_Coord'],window_length=100,split=4,mask_ratio=0.5):
        # self.data = pickle.load(
        #     open(file_path, "rb")
        # )
        self.data, self.labels = load_trajectory_data(root_dir, file_id, mode, traj_step_features, sc)
        #encoding methods
        # traj_embeddings = get_traj_embeddings(self.data, labels=None, verbose=False, seed=42)
        traj_transformer_embeddings = get_traj_transformer_embeddings(torch.tensor(self.data)).detach().numpy()
        # traj_ctle_embeddings=get_traj_ctle_embeddings(torch.tensor(self.data))
        shape = self.data.shape
        # 计算新的形状，将后面两个维度合并
        new_shape = (shape[0], shape[1] * shape[2])
        self.data = self.data.reshape(new_shape)

        self.data = traj_transformer_embeddings
        # self.data=traj_embeddings/4
        # 为了避免高斯噪声造成的影响过小，此处将原有的数值全部乘以20
        # self.data=traj_embeddings/8
        self.length = self.data.shape[0]

        self.mask_ratio = mask_ratio
        # self.test_data = pickle.load(
        #     open(test_path, "rb")
        # )
        # self.data = np.concatenate([self.data, self.test_data])

        # self.data = torch.Tensor(self.data)
        # 为了避免高斯噪声造成的影响过大，此处将原有的数值全部乘以20
        # self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - window_length))
        self.split = split

        # self.temp_dict = self.data.__getitem__(0)
        # self.observed_mask = self.temp_dict['observed_mask']
        #
        # self.mask_list = []
        #
        # for i in tqdm(range(0, 100)):
        #     self.mask_list.append(get_mask(self.observed_mask, mask_ratio=mask_ratio))

    def get_mask(self, observed_mask, strategy_type):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        if strategy_type == 0:
            # mask_ratio = self.mask_ratio

            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 == 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1
        else:
            # mask_ratio = 1 - self.mask_ratio
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 != 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1

        return mask

    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        if random.random() < 0.5:
            strategy_type = 0
        else:
            strategy_type = 1

        observed_data = self.data[
                        self.begin_indexes[item]:
                        self.begin_indexes[item] + self.window_length
                        ]
        observed_mask = np.ones_like(observed_data)
        gt_mask = self.get_mask(torch.from_numpy(observed_mask), strategy_type)
        timepoints = np.arange(self.window_length)
        return {
            "observed_data": torch.from_numpy(observed_data),
            "observed_mask": torch.from_numpy(observed_mask),
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "strategy_type": strategy_type
        }


        #         strategy_type = 0
        #     else:
        #         strategy_type = 1
        #     index_temp = np.random.randint((-length), -99)
        #     observed_data = self.data[
        #         self.begin_indexes[index_temp] :
        #            self.begin_indexes[index_temp] + self.window_length
        #     ]
        #     # observed_mask = torch.ones_like(observed_data)
        #     observed_mask=np.ones_like(observed_data)
        #     gt_mask = self.get_mask(torch.from_numpy(observed_mask), strategy_type)
        #     timepoints = np.arange(self.window_length)
        #
        #     temp_dict={
        #             "observed_data": torch.from_numpy(observed_data),
        #             "observed_mask": torch.from_numpy(observed_mask),
        #             "gt_mask": gt_mask,
        #             "timepoints": timepoints,
        #             "strategy_type": strategy_type
        #         }
class TestData(Dataset):

    def __init__(self, root_dir, file_id, mode, sc, submode='',get_label=False, split=4, window_split=1,
                 strategy=0, mask_list=[], traj_step_features=['X_Coord','Y_Coord'],window_length=10):
        self.strategy = strategy
        self.get_label = get_label

        self.mask_list = mask_list

        self.data, self.label = load_trajectory_data(root_dir, file_id, mode, traj_step_features, sc)
        traj_transformer_embeddings = get_traj_transformer_embeddings(torch.tensor(self.data)).detach().numpy()

        # traj_embeddings = get_traj_embeddings(self.data, labels=None, verbose=False, seed=42)
        # traj_embeddings = get_traj_transformer_embeddings(self.data)
        shape = self.data.shape
        # 计算新的形状，将后面两个维度合并
        new_shape = (shape[0], shape[1] * shape[2])
        self.data = self.data.reshape(new_shape)
        self.data = self.data
        # self.data=traj_embeddings/4
        # 为了避免高斯噪声造成的影响过小，此处将原有的数值全部乘以20
        # self.data=traj_embeddings/8
        self.data = traj_transformer_embeddings
        self.length = self.data.shape[0]


        self.label = torch.LongTensor(self.label)
        # self.data = np.concatenate([self.data, self.train_data])
        self.data = torch.Tensor(self.data)
        # self.data = self.data / 2
        # self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - self.window_length, self.window_length // window_split))
        self.mask_index = 0
        self.split = split

    def __len__(self):
        return len(self.begin_indexes)

    def get_mask(self, observed_mask):
        mask = torch.zeros_like(observed_mask)

        length = observed_mask.shape[0]

        if self.strategy == 0:
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 == 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1


        elif self.strategy == 1:
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 != 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1

        return mask




    def __getitem__(self, item):
        observed_data = self.data[
                        self.begin_indexes[item]:
                        self.begin_indexes[item] + self.window_length
                        ]
        observed_mask = torch.ones_like(observed_data)
        # print(f"item is {item}")
        gt_mask = self.get_mask(observed_mask)
        timepoints = np.arange(self.window_length)
        label = self.label[
            self.begin_indexes[item] :
           self.begin_indexes[item] + self.window_length
        ]


        if self.get_label:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                "label": label,
                'strategy_type': self.strategy
            }
        else:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                'strategy_type': self.strategy
            }

def get_mask(observed_mask, mask_ratio):
    mask = torch.zeros_like(observed_mask)

    original_mask_shape = mask.shape

    mask = mask.reshape(-1)
    total_index_list = list(range(len(mask)))

    selected_number = int(len(total_index_list) * mask_ratio)

    selected_index = sample(total_index_list, selected_number)

    selected_index = torch.LongTensor(selected_index)

    mask[selected_index] = 1

    mask = mask.reshape(original_mask_shape)

    return mask