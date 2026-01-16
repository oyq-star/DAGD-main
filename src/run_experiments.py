import json
import pickle

import numpy as np
import yaml

from loaddataset import TrainData, get_mask, TestData
from main_model import DGAD_Physio
from utils import train
from tqdm import tqdm
import os

num_worker = os.cpu_count()-1

# from loaddataset import TrajectoryDataset
from torch.utils.data import DataLoader
from utils import seed_all
import argparse
import time
import glob
import os

from experiment_configs import experiments_unsup_orig
from datetime import datetime

root_dir = './datasets/files_valid/'  # content is 'valid'


def run(root_dir):

    # train_file_1 = pickle.load(
    #     open(f"datasets/data_pkl/amazon_windows_length100_checkpoint_trn49.pkl", "rb")
    # )



    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_all = [experiments_unsup_orig]
    experiment_runs=experiments_unsup_orig
    for experiment_runs in exp_all:
        for exp in experiment_runs:
            if exp['model']['model_type'] != 'DGAD': continue
            exp_done = glob.glob(f"./results/*_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_*.txt")
            if len(exp_done)>0:
                print(f"{exp_done}")
                continue
            ds_trn_id, ds_vld_id, ds_tst_id = exp['setting']['ds_train'], exp['setting']['ds_valid'], exp['setting']['ds_test']

            traj_step_features=exp['setting']['ds']['trj_step_feat']

            seq_len = exp['setting']['ds']['input_dims']
            run_history = {}

            exp_setting=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{exp['scaler']}_trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}"
            print(exp_setting)

            ##############################################################

            dataset_name=exp['setting']['ds']['dataset_name'] #'brightkite' #'synthetic' # 'amazon' # 'dbcargo'

            model_path="./temp/"
            os.makedirs(model_path, exist_ok=True)
            model_name=f"{exp['model']['model_type']}_checkpoint_trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}.pt"



            if 'yn' in dataset_name:
                bs=256
                lr=1e-03
                wd=0.
                drp=0.
                seq_len=72
                seq_len_dk=seq_len #72
                num_heads=12
                feature_dim = 2*seq_len
                # feature_dim = 32

            if 'amazon' in dataset_name:
                bs=256
                lr=1e-03
                wd=0.
                drp=0.
                seq_len=72
                seq_len_dk=seq_len #72
                num_heads=12
                feature_dim = 2*seq_len

            if 'brightkite' in dataset_name:
                bs=256
                lr=1e-04
                wd=0.
                drp=0.
                seq_len=500
                seq_len_dk=seq_len #500
                num_heads=8
                feature_dim = 2*seq_len


            if 'dbcargo' in dataset_name:
                bs=256
                lr=(1e-02, 1e-03)['RU' in model_name]
                wd=0.
                drp=0.
                seq_len=72
                seq_len_dk=seq_len #72
                num_heads=12
                feature_dim=2*seq_len
                # feature_dim = 32


            num_layers=(4,2)['RU' in model_name]
            clip_value = (None, 5)['RU' in model_name]
            opt='RAdam'
            seg_len=2

            print(f"lr:{lr} bs:{bs} wd:{wd} drp:{drp} ds:{dataset_name} seq_len:{seq_len} seq_len_dk:{seq_len_dk} seg_len:{seg_len}")

            for seed in [0]:

                t1_ = time.time()
                print(f" ################ SEED {seed} - trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id} ################")

                seed_all(seed)

                args = argparse.Namespace(d_step_feat=len(traj_step_features), d_inp_size=seq_len, d_inp_embed_size=16,
                                          d_k=seq_len_dk, d_ffn_embed_size=2048, enc_layers=num_layers, num_heads=num_heads, num_nonlinearity_layers=2,
                                          k=None, sc='standard', lr=lr, batch_size=bs, epochs=150, clip_value=clip_value,
                                          pat_es=20, #5,
                                          pat_sched=10,
                                          wd=wd, min_lr=1e-6, drop=drp, valid_loss_only=True,
                                          winit_orig=None, progressive_training=0,
                                          model_file_path=f"{model_path}{model_name}",
                                          seg_len=seg_len, padding_mode='circular', padding_val=None, verbose=False, log_engs=False, bas_phase='tst',
                                          opt=opt, temperature=10, lambda_parm=1, d_hidden_size=seq_len,ratio=0.7,window_length=5,
                                          mask_ratio=0.5,split=2,window_split=1,device='cuda:0',config='base.yaml',dataset=dataset_name,seed=seed,ds_trn_id=ds_trn_id,ds_vld_id=ds_vld_id,ds_tst_id=ds_tst_id
                                          )
                ##dbcargo: window_length=5 split=1
                #dbcargo: window_length=10
                path = "config/" + 'base.yaml'
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                print("-------------")
                print(args)
                print("-------------")

                print(f"{model_name} running...")
                for training_epoch in range(0, 1):
                    unconditional=True
                    diffusion_step=50
                    path = "config/" + args.config
                    with open(path, "r") as f:
                        config = yaml.safe_load(f)

                    config["model"]["is_unconditional"] = unconditional

                    config["diffusion"]["num_steps"] = diffusion_step
                    print(json.dumps(config, indent=4))

                    foldername = f"./train_result/save{training_epoch}/" + f"{root_dir[:-1]}_" +  f"{dataset_name}_origin_0_"+f"{ds_trn_id}" +"_unconditional:" + str(
                        unconditional) + "_split:" + str(
                        args.split) + "_diffusion_step:" + str(diffusion_step) + "/"
                    print('model folder:', foldername)
                    if not os.path.exists(foldername):
                        os.makedirs(foldername)
                    with open(foldername + "config.json", "w") as f:
                        json.dump(config, f, indent=4)

                    train_data = TrainData(root_dir, file_id=ds_trn_id, mode='train', sc=args.sc, submode='', traj_step_features=traj_step_features, window_length=args.window_length,split=args.split, mask_ratio=args.mask_ratio)
                    valid_data = TrainData(root_dir, file_id=ds_vld_id, mode='valid', sc=args.sc, submode='', traj_step_features=traj_step_features,window_length=args.window_length, split=args.split, mask_ratio=args.mask_ratio)
                    try:
                        # 以二进制写入模式打开文件
                        with open(f"./datasets/data_pkl/{dataset_name}_windows_length{args.window_length}_checkpoint_trn{ds_trn_id}_origin_0.pkl", 'wb') as file:
                            # 使用 pickle.dump 方法将数据保存到文件中
                            pickle.dump(train_data, file)
                        print("数据已成功保存为 train_data.pkl")
                        with open(f"./datasets/data_pkl/{dataset_name}_windows_length{args.window_length}_checkpoint_vld{ds_vld_id}_origin_0.pkl", 'wb') as file:
                            # 使用 pickle.dump 方法将数据保存到文件中
                            pickle.dump(valid_data, file)
                        print("数据已成功保存为 valid_data.pkl")
                    except Exception as e:
                        print(f"保存文件时出现错误: {e}")


                    index_temp = np.random.randint(0, train_data.length-train_data.window_length)
                    temp_dict = train_data.__getitem__(index_temp)
                    observed_mask = temp_dict['observed_mask']

                    mask_list = []

                    # for i in tqdm(range(0, 100)):
                    for i in tqdm(range(0, train_data.window_length)):
                        mask_list.append(get_mask(observed_mask, mask_ratio=args.mask_ratio))
                    test_data_strategy_1 = TestData(root_dir, file_id=ds_tst_id, mode='test', sc=args.sc, submode='', traj_step_features=traj_step_features,window_length=args.window_length, split=args.split, window_split=args.window_split,
                                                    strategy=0,  mask_list=mask_list)
                    test_data_strategy_2 = TestData(root_dir, file_id=ds_tst_id, mode='test', sc=args.sc, submode='', traj_step_features=traj_step_features,window_length=args.window_length, split=args.split, window_split=args.window_split,
                                                    strategy=1,  mask_list=mask_list)
                    try:
                        # 以二进制写入模式打开文件
                        with open(f"./datasets/data_pkl/{dataset_name}_windows_length{args.window_length}_checkpoint_tst{ds_tst_id}_1_origin_0.pkl", 'wb') as file:
                            # 使用 pickle.dump 方法将数据保存到文件中
                            pickle.dump(test_data_strategy_1, file)
                        print("数据已成功保存为 test1_data.pkl")
                        with open(f"./datasets/data_pkl/{dataset_name}_windows_length{args.window_length}_checkpoint_tst{ds_tst_id}_2_origin_0.pkl", 'wb') as file:
                            # 使用 pickle.dump 方法将数据保存到文件中
                            pickle.dump(test_data_strategy_2, file)
                        print("数据已成功保存为 test2_data.pkl")
                    except Exception as e:
                        print(f"保存文件时出现错误: {e}")


                    train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
                    # valid_loader = DataLoader(valid_data, batch_size=12, shuffle=True)
                    valid_loader = DataLoader(valid_data, batch_size=12, shuffle=True) #针对dbcargo

                    test_loader1 = DataLoader(test_data_strategy_1, batch_size=12)
                    test_loader2 = DataLoader(test_data_strategy_2, batch_size=12)


                    model_diff=DGAD_Physio(config, args.device,target_dim=feature_dim,ratio = args.ratio).to(args.device)
                    print('DGAD_Physio_sum: ',sum(p.numel() for p in model_diff.parameters()))



                    train(
                        model_diff,
                        config["train"],
                        train_loader,
                        valid_loader=valid_loader,
                        foldername=foldername,
                        test_loader1=test_loader1,
                        test_loader2=test_loader2
                    )
    #                 dict_hist, best_model = train_model(model, dl_trn, dl_vld, args, dict_hist)
    #                 dict_hist = test_model(best_model, dl_tst, args, dict_hist)
    #
    #                 model = None
    #                 gc.collect(generation=0)
    #
    #                 seed_test_auroc = dict_hist['tst']['auroc'][-1]
    #                 seed_test_auprc = dict_hist['tst']['auprc'][-1]
    #
    #                 t2_ = time.time()
    #                 print(f" ################ SEED {seed} - trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}: AUROC: {round(seed_test_auroc,prec)} AUPRC: {round(seed_test_auprc, prec)} duration: {round((t2_-t1_)/60.,1)}min ################")
    #
    #                 seed_performances['auroc'].append(seed_test_auroc)
    #                 seed_performances['auprc'].append(seed_test_auprc)
    #
    #                 run_history[seed] = dict_hist
    #
    #         avg_auroc = np.array(seed_performances['auroc']).mean()
    #         avg_auprc = np.array(seed_performances['auprc']).mean()
    #         std_auroc = np.array(seed_performances['auroc']).std()
    #         std_auprc = np.array(seed_performances['auprc']).std()
    #
    #         print(f"AVG AUROC: {round(avg_auroc,prec)}+-{round(std_auroc,3)} AVG AUPRC: {round(avg_auprc, prec)}+-{round(std_auprc,3)}")
    #
    #
    #         ##############################################################
    #
    #         #result_filename=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{round(avg_auroc,3)}_{round(avg_auprc,3)}_{exp['scaler']}"
    #         result_filename=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{round(avg_auroc,3)}_{round(avg_auprc,3)}"
    #         print(f"### {result_filename} ({len(seeds)} seeds): AUROC: {round(avg_auroc, prec)}+/-{round(std_auroc,3)} AUPRC: {round(avg_auprc, prec)}+/-{round(std_auprc,3)}")
    #
    #         perf = {'auroc': avg_auroc, 'auroc_std': std_auroc, 'auprc': avg_auprc, 'auprc_std': std_auprc}
    #         result_filename_all="./results/{result_filename}.txt"
    #         if not os.path.exists(result_filename_all):
    #             os.makedirs(os.path.dirname(result_filename_all), exist_ok=True)
    #         with open(f"./results/{result_filename}.txt", "w") as f:
    #             f.write(str({'summary': perf, 'hist': run_history, 'args': args, 'exp': exp}))
    #
    # print(f"RUNS {ts} FINISHED")



