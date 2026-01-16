import gc

import numpy as np

from diff_models import diff_DGAD
import torch
import torch.nn as nn
import copy

from delected.input_representation import InputEmbedding, PositionalEncoding, BERTInputRepresentation
from delected.transformer_encoder import TransformerEncoderBlockLayer
class DGAD_base(nn.Module):
    def __init__(self, target_dim, config, device,ratio = 0.7):
        super().__init__()


        self.device = device
        self.ratio = ratio
        self.target_dim = target_dim

        self.ddim_eta = 1
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.is_unconditional=True
        # self.emb_time_dim = 128
        # self.emb_feature_dim = config["model"]["featureemb"]

        self.target_strategy = 'random'
        print("unconditional is")
        print(self.is_unconditional)
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_DGAD(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        # cumprod函数表示将之前的alpha连乘。这里的self.alpha实际上就是\overline \alpha
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask,ratio = 0.7):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) #(b, *)
        for i in range(len(observed_mask)):
            # sample_ratio = np.random.rand()  # missing ratio
            sample_ratio = ratio  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            # 选择num_masked个数字，让它等于-1
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask,ratio=self.ratio)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) #12*100*128  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)#12*100*38*128
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)#38*16
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)#12*100*38*16

        side_info = torch.cat([time_embed, feature_embed], dim=-1) #12*100*38*144 # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)@12*144*38*100

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type=strategy_type, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        # 当为unconditional的时候，total_input就是noisy data
        predicted = self.diffmodel(total_input, side_info, t, strategy_type)  # (B,K,L)

        # 此处的condition mask全部为0
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples,strategy_type):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device),strategy_type)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                # print("alpha_hat t is")
                # print(self.alpha_hat[t])
                # print("shape of alpha hat t is ")
                # print( self.alpha_hat[t].shape)
                # 注意一下，这里的alpha_hat以及alpha和DDPM论文当中的alpha是正好相反的。
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                    # print("shape of current samples is")
                    # print(current_sample.shape)


            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


    def ddim_impute(self, observed_data, cond_mask, side_info, n_samples,ddim_eta=1,ddim_steps=10):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)


        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    # 这里的noisy_cond_history就是对整个数据片上的所有数据进行了加噪
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            ddim_timesteps = ddim_steps
            c = self.num_steps // ddim_timesteps
            ddim_timesteps_sequence = np.asarray(list(range(0, self.num_steps, c)))
            ddim_timesteps_previous_sequence = np.append(
                np.array([0]) , ddim_timesteps_sequence[: -1]
            )

            for step_number in range(ddim_timesteps - 1 , -1, -1):
                t = ddim_timesteps_sequence[step_number]
                previous_t =  ddim_timesteps_previous_sequence[step_number]

                at = torch.tensor(self.alpha[t]).to(self.device)
                at_next = torch.tensor(self.alpha[previous_t]).to(self.device)

                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                xt = diff_input
                et = self.diffmodel(xt, side_info, torch.tensor([t]).to(self.device))
                x0_t = (current_sample - et * (1 - at).sqrt()) / at.sqrt()

                c1 = (
                        ddim_eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                current_sample = at_next.sqrt() * x0_t + c1 * torch.randn_like(current_sample) + c2 * et

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


    def get_middle_impute_value(self, observed_data, cond_mask, side_info, n_samples, strategy_type):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_middle_samples = torch.zeros(B,self.num_steps, K, L)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in (range(self.num_steps - 1, -1, -1)):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device),strategy_type)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

                imputed_middle_samples[:,t] = current_sample.detach()

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples, imputed_middle_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            strategy_type
        ) = self.process_data(batch)
        # print("observed data shape is")
        # print(observed_data.shape)
        # print("observed mask shape is")
        # print(observed_mask.shape)
        # print("observed tp is")
        # print(observed_tp)
        # 强制使用0作为cond_mask
        self.target_strategy = "random"
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask,ratio=self.ratio)
            #
            # cond_mask = torch.zeros_like(observed_mask)
            # cond_mask = self.get_random_mask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)# # (B,*,K,L)@12*144*38*100

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type = strategy_type)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,#时间信息
            gt_mask,
            _,
            cut_length,
            strategy_type
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            print(f"strategy type in evaluate is {strategy_type}")
            samples = self.impute(observed_data, cond_mask, side_info, n_samples, strategy_type)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        # 此处target_mask给的是那些待预测的点
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def get_middle_evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples,imputed_middle_samples = self.get_middle_impute_value(observed_data, cond_mask, side_info, n_samples, strategy_type)

            print("shape of imputed middle samples is")
            print(imputed_middle_samples.shape)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp, imputed_middle_samples

    def ddim_evaluate(self, batch, n_samples,ddim_eta=1,ddim_steps=10):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.ddim_impute(observed_data, cond_mask, side_info, n_samples,ddim_eta=ddim_eta,ddim_steps=ddim_steps)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

class CustomTaskHead(nn.Module): # for GAD on Trajectories
           
    def __init__(self, d_model, d_inp_embed_size, num_nonlinearity_layers=2, dropr=0.): #v4
        super(CustomTaskHead, self).__init__()
        
        self.d_model = d_model
        self.d_inp_embed_size = d_inp_embed_size
        self.dropr = dropr
        self.num_nonlinearity_layers = num_nonlinearity_layers
        
        
        self.task_nonlinearity_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_model, d_model),   # v4: torch.Size([256, 72, 16]) 
                nn.ReLU()                      
            ) for _ in range(0, num_nonlinearity_layers)
        ])
        
        self.prj_inp_embed = nn.Linear(d_inp_embed_size, 1) 
        self.prj_d_model = nn.Linear(d_model, 1) # v4
        self.dropout = nn.Dropout(dropr)
        
    def forward(self, x):
        
        m = x.shape[0] # bs        
        x_ = x 
        
        for lidx in range(0, self.num_nonlinearity_layers):
            
            # x_ - torch.Size([256, 16, 72])
            x_ = self.task_nonlinearity_layers[lidx](x_)         # torch.Size([256, 16, 72])
            
        x1 = x_.view((m, self.d_model, self.d_inp_embed_size))   # torch.Size([256, 72, 16]) # v4
        
        x2 = self.prj_inp_embed(x1)                              # torch.Size([256, 72, 1])
        
        x3 = x2.view((m, 1, self.d_model))                       # torch.Size([256, 1, 72]) # v4
        
        x4 = self.prj_d_model(x3)                                # torch.Size([256, 1, 1])  # v4
        x5 = self.dropout(x4)                                    
        
        return x5

def weight_reinitialization(model):
    initru=0.1
    for name, p in model.named_parameters():
        if 'prj' in name or 'task' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(p.data)
        elif 'weight_ih' in name:
            None
            #nn.init.xavier_uniform_(p.data)
        elif 'weight_hh' in name:
            None
            #nn.init.uniform_(p.data, a=-initru, b=initru) 
            #nn.init.xavier_uniform_(p.data)               
        if 'bias' in name:
            p.data.fill_(0)
            

def freeze_weights(model, param_name, unfreeze, verbose=False):
    for a, b in model.named_parameters():
        if param_name in a:
            b.requires_grad=unfreeze
            if verbose: print(b)
                

class GADFormer(DGAD_base,nn.Module):

    def __init__(self, d_step_feat, d_inp_size, d_inp_embed_size, d_k, d_ffn_embed_size, num_heads, num_layers=4, num_nonlinearity_layers=2, dropr=0.0, save_attn=True, winit_orig=True, progressive_training=None, seg_len=1, padding_mode='circular', padding_val=None, verbose=False, log_engs=False, bas_phase='tst',target_dim=38,ratio=0.7,device='cuda:0',config=None):
    
        super(GADFormer, self).__init__(target_dim, config, device)
        
        self.d_step_feat = d_step_feat
        self.d_inp_size = d_inp_size
        self.d_model = d_inp_size    # for sequences d_model has to be d_inp_size (=seq_len), otherwise InputEmbedding needs to conduct an additional projection. (v4)
        self.d_k = d_k
        self.d_inp_embed_size = d_inp_embed_size
        self.d_ffn_embed_size = d_ffn_embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.log_engs = log_engs
        self.energies = {'trn': {}, 'vld': {}}
        bl_dct = {bl_idx: [] for bl_idx in range(0, num_layers)}
        self._hgrps_all_dct = {'trn': copy.deepcopy(bl_dct), 'vld': copy.deepcopy(bl_dct), 'tst': copy.deepcopy(bl_dct)}
        self.progressive_training = progressive_training
        self.seg_len = seg_len
        self.padding_mode = padding_mode
        self.padding_val = padding_val
        self.save_attn = save_attn
        self.bas_phase = bas_phase
        self.verbose = verbose
        self.unfrozen = False
        
        
        self.input_embedding = InputEmbedding(d_inp_size, d_step_feat, d_inp_embed_size) 
        
        self.positional_encoding = PositionalEncoding(d_model=d_inp_size, d_inp_embed_size=d_inp_embed_size) #v4
        
        self.bert_input_embedding = BERTInputRepresentation(self.input_embedding, self.positional_encoding, self.d_inp_size, seg_len, padding_mode, padding_val, verbose) 
        
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlockLayer(d_inp_size, d_inp_embed_size, self.d_model, self.d_k, d_ffn_embed_size, num_heads, dropr, save_attn, winit_orig, log_engs) for _ in range(0, num_layers)] # v4
        )
        
        self.custom_task_head = CustomTaskHead(self.d_model, d_inp_embed_size, num_nonlinearity_layers, dropr) #v4
        
        if self.progressive_training is not None:
            freeze_weights(self, 'custom_task_head', unfreeze=False)
            print("custom_task_head frozen")

        self.target_dim=d_inp_embed_size
        self.ratio=ratio
        self.config = config
        self.device = 'cuda:0'

    def reset_attn(self): 
        
        del self._hgrps_all_dct
        gc.collect(generation=0)
        
        bl_dct = {bl_idx: [] for bl_idx in range(0, self.num_layers)}
        self._hgrps_all_dct = {'trn': copy.deepcopy(bl_dct), 'vld': copy.deepcopy(bl_dct), 'tst': copy.deepcopy(bl_dct)}
                
    
    def log_energies(self, key, ep, x):
        
        if self.log_engs:
            phase = ('vld','trn')[self.training]        
            if key not in self.energies[phase].keys():
                self.energies[phase][key] = {}
            if ep not in self.energies[phase][key].keys():                
                self.energies[phase][key][ep] = []    
            self.energies[phase][key][ep].append(x)
        
    
    def forward(self, x, ep, es=None, phase='trn'):
        
        # x - torch.Size([256, 72, 2]) 
        m = x.shape[0]
                
        x1 = self.bert_input_embedding.forward(x)
        #x1 - torch.Size([256, 72, 16])
        x_in = x1
        for i, enc_block in enumerate(self.encoder):
            
            _hgprs_all_ = (None, self._hgrps_all_dct[phase][i])[phase=='trn' or self.bas_phase==phase]
            
            x2 = enc_block.forward(x_in, ep, i, _hgprs_all_)
                
            x_in = x2                                  # torch.Size([256, d_model, d_imp_embed_size]) , torch.Size([256, 72, 16])    
            
        x3 = x_in                                      # v4: torch.Size([256, 72, 16])
        
        x4 = x3.view(m, self.d_inp_embed_size, self.d_model) # v4: torch.Size([256, 16, 72])
        
        if self.training:
            if es is None or (self.progressive_training is not None and es.es_score_min != np.Inf and es.counter > self.progressive_training) or self.unfrozen:
                freeze_weights(self, 'custom_task_head', unfreeze=True)
                self.unfrozen = True
         
        x5 = self.custom_task_head(x4)                # torch.Size([256, 1, 1])
        
        
        if self.log_engs: self.log_energies(f"enc{i}ct", ep, x5.clone().detach().cpu().numpy())
        
        x6 = torch.sigmoid(x5)                        # torch.Size([256, 1, 1])
        
        if self.log_engs: self.log_energies(f"enc{i}sio", ep, x6.clone().detach().cpu().numpy())
        
        x7 = x6.squeeze(2)                            # torch.Size([256, 1])
        
        return x7
    
            

class GRUBaseline(nn.Module):
    
    def __init__(self, d_step_feat, d_inp_size, d_inp_embed_size, d_hidden_size, batch_size, num_layers=4, num_nonlinearity_layers=2, dropr=0.0, 
                 progressive_training=None, verbose=False):
        super(GRUBaseline, self).__init__()
        
        self.d_inp_size = d_inp_size
        self.d_inp_embed_size = d_inp_embed_size
        self.num_layers = num_layers
        self.d_hidden_size = d_hidden_size
        self.progressive_training = progressive_training
        self.dropr = dropr
        self.bidirectional = False
        self.D = (1,2)[self.bidirectional]
        self.verbose = verbose
        self.unfrozen = False
        self.save_attn = False
        
        self.input_embedding = InputEmbedding(d_inp_size, d_step_feat, d_inp_embed_size) # v2
        
        #self.ih0 = torch.zeros(self.D * self.num_layers, batch_size, self.d_hidden_size).to(x.device) # shape (D * num_layers, H_out) or (D*num_layers, N, H_out) 
        
        self.gru_baseline_net = nn.GRU(d_inp_embed_size, d_hidden_size, num_layers, bidirectional=self.bidirectional, batch_first=True, bias=True, dropout=dropr)
        self.prj_gru_to_task = nn.Linear(d_hidden_size, d_inp_embed_size)
        
        self.custom_task_head = CustomTaskHead(d_inp_size, d_inp_embed_size, num_nonlinearity_layers, dropr) #v4
        
        if self.progressive_training is not None:
            freeze_weights(self, 'custom_task_head', unfreeze=False)
            print("custom_task_head frozen")
            
        self.apply(weight_reinitialization)
        
    
    def forward(self, x, ep, es=None, phase='trn'):
        
        m = x.shape[0]
        
        if self.verbose: print(f"gruou  x {x.shape}")       # torch.Size([256, 72, 2])
        
        x1 = self.input_embedding(x)                        # torch.Size([256, 72, 16])
        if self.verbose: print(f"gruou  x1 {x1.shape}")
        
        self.ih0 = torch.zeros(self.D * self.num_layers, m, self.d_hidden_size).to(x.device) # shape (D * num_layers, H_out) or (D*num_layers, N, H_out) 
        
        x2, hn = self.gru_baseline_net(x1, self.ih0[:,:m,:].to(x1.device))
        #x2, hn = self.gru_baseline_net(x1)
        
        if self.verbose: print(f"gruou x2 {x2.shape}")      # torch.Size([256, 72, 50])
        if self.verbose: print(f"gruou hn {hn.shape}")                                     # torch.Size([4, 256, 50])
        
        
        x3 = self.prj_gru_to_task(x2)
        if self.verbose: print(f"gruou x3 {x3.shape}")
        
        x4 = x3.view(m, self.d_inp_embed_size, self.d_inp_size)
        if self.verbose: print(f"gruou x4 {x4.shape}")
        
        if self.training:
            if es is None or (self.progressive_training is not None and es.es_score_min != np.Inf and es.counter > self.progressive_training) or self.unfrozen:
                freeze_weights(self, 'custom_task_head', unfreeze=True)
                self.unfrozen = True
                
        x5 = self.custom_task_head(x4)
        if self.verbose: print(f"gruou x5 {x5.shape}")
        
        x6 = torch.sigmoid(x5)        
        if self.verbose: print(f"gruou x6 {x6.shape}")
        
        x7 = x6.squeeze(2)      
        if self.verbose: print(f"gruou x7 {x7.shape}")
        
        return x7
    
