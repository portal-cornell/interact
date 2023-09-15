''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch_dct as dct
import numpy as np
from MRT.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

JOINT_AMT = 9
BOB_HUMAN_JOINT_AMT = 9
BOB_ROBOT_JOINT_AMT = 9

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        # self.register_buffer('pos_table3', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x,n_person):
        p=self.pos_table[:,:x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        # if x.shape[1]==135:
        #     p=self.pos_table3[:, :int(x.shape[1]/n_person)].clone().detach()
        #     p=p.repeat(1,n_person,1)
        # else:
        p=self.pos_table2[:, :int(x.shape[1]/n_person)].clone().detach()
        p=p.repeat(1,n_person,1)
        return x + p


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, device='cuda'):

        super().__init__()
        self.position_embeddings = nn.Embedding(n_position, d_model)
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device
    def forward(self, src_seq,n_person, src_mask, return_attns=False, global_feature=False):
        
        enc_slf_attn_list = []
        # -- Forward
        #src_seq = self.layer_norm(src_seq)
        if global_feature:
            enc_output = self.dropout(self.position_enc.forward2(src_seq,n_person))
            #enc_output = self.dropout(src_seq)
        else:
            enc_output = self.dropout(self.position_enc(src_seq,n_person))
        #enc_output = self.layer_norm(enc_output)
        #enc_output=self.dropout(src_seq+position_embeddings)
        #enc_output = self.dropout(self.layer_norm(enc_output))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list


        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,device='cuda'):

        super().__init__()

        #self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)
        # import pdb;pdb.set_trace()
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

class IntentInformedHRForecaster(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, 
            dropout=0.2, n_position=100, 
            conditional_forecaster=False,
            alice_joints_num = 9,
            bob_joints_list = None,
            bob_joints_num = 9,
            one_hist = False,
            robot_joints_list = None,
            robot_joints_num = 2,
            align_rep = False,
            device='cuda'):
    
        super().__init__()
        
        self.hh = IntentInformedForecaster(
            src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, 
            dropout=dropout, n_position=n_position, 
            conditional_forecaster=conditional_forecaster,
            alice_joints_num = alice_joints_num,
            bob_joints_list = bob_joints_list,
            bob_joints_num = bob_joints_num,
            one_hist = one_hist,
            device=device
        )

        self.robot_joint_indices = None
        if robot_joints_list is not None:
            joint_idx_size = len(robot_joints_list) * 3
            robot_joint_indices = np.zeros(joint_idx_size, dtype=int)
            for i, value in enumerate(robot_joints_list):
                for j in range(3):
                    robot_joint_indices[i*3+j] = value*3 + j
            self.robot_joint_indices = robot_joint_indices

        self.robot_global_hist_encoder=nn.Linear(robot_joints_num*3,d_model) 

        self.robot_global_future_encoder=nn.Linear(robot_joints_num*3,d_model) 

        self.robot_joints_num = robot_joints_num

        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.align_rep = align_rep

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
    
    def forward(self, 
            alice_hist, 
            bob_hist, 
            bob_future,
            robot_hist,
            robot_future,
            add_spe=True):
        ### This should be zero
        alice_current_pos = alice_hist[:, -1, :].unsqueeze(1)
        
        ### Index out bob's relevant joints
        if self.hh.bob_joint_indices is not None:
            bob_future = bob_future[:,:,self.hh.bob_joint_indices]
            bob_hist = bob_hist[:,:,self.hh.bob_joint_indices]

        ### Intent informed forecasting only cares about Bob's position at final timestep
        bob_future = bob_future[:,-1:]
        robot_future = robot_future[:, -1:]
        
        ### local history encoding
        alice_displacement = alice_hist[:,1:alice_hist.shape[1],:]-alice_hist[:,:alice_hist.shape[1]-1,:]                 
        alice_displacement_dct = dct.dct(alice_displacement)
        alice_local_enc = self.hh.alice_local_hist_encoder(alice_displacement_dct)
        alice_local_output, *_ = self.hh.encoder(alice_local_enc, 1, None)

        ### global history encoding
        alice_global_enc = self.hh.alice_global_hist_encoder(alice_hist)
        if not self.hh.one_hist:
            bob_global_enc = self.hh.bob_global_hist_encoder(bob_hist)
            robot_global_enc = self.robot_global_hist_encoder(robot_hist)
            global_enc = torch.cat([alice_global_enc, robot_global_enc],dim=1)
        else:
            global_enc = alice_global_enc
        global_output, *_ = self.hh.encoder_global(global_enc,
                    1 if self.hh.one_hist else 2, 
                    src_mask = None, 
                    global_feature=True)

        ### conditional future encoder
        if not self.hh.conditional_forecaster:
            bob_future.fill_(0)
        bob_cond_future_enc = self.hh.bob_global_future_encoder(bob_future)
        robot_cond_future_enc = self.robot_global_future_encoder(robot_future)

        spe = 0
        if add_spe:
            alice_spe = torch.norm(alice_hist-alice_hist[:, -1].unsqueeze(1), dim=-1)
            robot_spe = torch.norm(robot_hist-alice_hist[:, -1, self.robot_joint_indices].unsqueeze(1), dim=-1)
            spe = torch.exp(-torch.cat([alice_spe, robot_spe] if not self.hh.one_hist else [alice_spe], dim=1)).unsqueeze(2)

        encoder_output = torch.cat([alice_local_output, global_output+spe], dim=1)

        dec_output, dec_attention, *_ = self.hh.decoder(robot_cond_future_enc, None, encoder_output, None)
        dec_output = self.hh.decoder_linear(dec_output)
        dec_output = torch.permute(dec_output, (0,2,1)) # (batch size, 1, d_model) -> (batch size, d_model, 1)
        dec_output = self.hh.linear_proj_to_forecast(dec_output) # (batch size, d_model, 1) -> (batch size, d_model, 15)
        dec_output = torch.permute(dec_output, (0,2,1)) # (batch size, d_model, 15) -> (batch size, 15, d_model)
        alice_forecasts_dct = self.hh.forecast_head(dec_output)
        alice_forecasts_displacments = dct.idct(alice_forecasts_dct)

        alice_forecasts = torch.cumsum(alice_forecasts_displacments, dim=1)

        align_loss = 0
        if self.align_rep:
            if not self.hh.one_hist:
                cos_dist_hist = 1-self.cosine_sim(bob_global_enc, robot_global_enc)
                align_loss += torch.mean(cos_dist_hist.reshape(-1, 1))
            if self.hh.conditional_forecaster:
                cos_dist_fut = 1-self.cosine_sim(bob_cond_future_enc, robot_cond_future_enc)
                align_loss += torch.mean(cos_dist_fut.reshape(-1, 1))

        return alice_forecasts, align_loss
    
class IntentInformedForecaster(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, 
            dropout=0.2, n_position=100, 
            conditional_forecaster=False,
            alice_joints_num = 9,
            bob_joints_list = None,
            bob_joints_num = 9,
            one_hist = False,
            device='cuda'):
    
        super().__init__()
        self.conditional_forecaster = conditional_forecaster
        self.one_hist = one_hist
        self.bob_joint_indices = None
        if bob_joints_list is not None:
            joint_idx_size = len(bob_joints_list) * 3
            bob_joint_indices = np.zeros(joint_idx_size, dtype=int)
            for i, value in enumerate(bob_joints_list):
                for j in range(3):
                    bob_joint_indices[i*3+j] = value*3 + j
            self.bob_joint_indices = bob_joint_indices
        
        self.device=device
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.alice_local_hist_encoder=nn.Linear(alice_joints_num*3,d_model) 

        self.alice_global_hist_encoder=nn.Linear(alice_joints_num*3,d_model)
        self.bob_global_hist_encoder=nn.Linear(bob_joints_num*3,d_model) 

        self.bob_global_future_encoder=nn.Linear(bob_joints_num*3,d_model) 
        
        self.decoder_linear = nn.Linear(d_model,d_model)
        self.linear_proj_to_forecast = nn.Linear(1,15)
        self.forecast_head=nn.Linear(d_model,alice_joints_num*3)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
    
    def forward(self, 
            alice_hist, 
            bob_hist, 
            bob_future,
            add_spe=True, 
            bob_is_robot=False):
        ### This should be zero
        alice_current_pos = alice_hist[:, -1, :].unsqueeze(1)
        
        ### Index out bob's relevant joints
        if self.bob_joint_indices is not None:
            bob_future = bob_future[:,:,self.bob_joint_indices]
            bob_hist = bob_hist[:,:,self.bob_joint_indices]

        ### Intent informed forecasting only cares about Bob's position at final timestep
        bob_future = bob_future[:,-1:]
        
        ### local history encoding
        alice_displacement = alice_hist[:,1:alice_hist.shape[1],:]-alice_hist[:,:alice_hist.shape[1]-1,:]                 
        alice_displacement_dct = dct.dct(alice_displacement)
        alice_local_enc = self.alice_local_hist_encoder(alice_displacement_dct)
        alice_local_output, *_ = self.encoder(alice_local_enc, 1, None)

        ### global history encoding
        alice_global_enc = self.alice_global_hist_encoder(alice_hist)
        if not self.one_hist:
            bob_global_enc = self.bob_global_hist_encoder(bob_hist)
            global_enc = torch.cat([alice_global_enc, bob_global_enc],dim=1)
        else:
            global_enc = alice_global_enc
        global_output, *_ = self.encoder_global(global_enc,
                    1 if self.one_hist else 2, 
                    src_mask = None, 
                    global_feature=True)

        ### conditional future encoder
        if not self.conditional_forecaster:
            bob_future.fill_(0)
        bob_cond_future_enc = self.bob_global_future_encoder(bob_future)

        spe = 0
        if add_spe:
            alice_spe = torch.norm(alice_hist-alice_hist[:, -1].unsqueeze(1), dim=-1)
            bob_spe = torch.norm(bob_hist-alice_hist[:, -1, self.bob_joint_indices].unsqueeze(1), dim=-1)
            spe = torch.exp(-torch.cat([alice_spe, bob_spe] if not self.one_hist else [alice_spe], dim=1)).unsqueeze(2)

        encoder_output = torch.cat([alice_local_output, global_output+spe], dim=1)

        dec_output, dec_attention, *_ = self.decoder(bob_cond_future_enc, None, encoder_output, None)
        dec_output = self.decoder_linear(dec_output)
        dec_output = torch.permute(dec_output, (0,2,1)) # (batch size, 1, d_model) -> (batch size, d_model, 1)
        dec_output = self.linear_proj_to_forecast(dec_output) # (batch size, d_model, 1) -> (batch size, d_model, 15)
        dec_output = torch.permute(dec_output, (0,2,1)) # (batch size, d_model, 15) -> (batch size, 15, d_model)
        alice_forecasts_dct = self.forecast_head(dec_output)
        alice_forecasts_displacments = dct.idct(alice_forecasts_dct)

        alice_forecasts = torch.cumsum(alice_forecasts_displacments, dim=1)

        return alice_forecasts

class ConditionalForecaster(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, 
            dropout=0.2, n_position=100, 
            conditional_forecaster=False,
            alice_joints_num = 9,
            bob_joints_list = None,
            bob_joints_num = 9,
            one_hist = False,
            device='cuda'):
    
        super().__init__()
        self.conditional_forecaster = conditional_forecaster
        self.one_hist = one_hist
        self.bob_joint_indices = None
        if bob_joints_list is not None:
            joint_idx_size = len(bob_joints_list) * 3
            bob_joint_indices = np.zeros(joint_idx_size, dtype=int)
            for i, value in enumerate(bob_joints_list):
                for j in range(3):
                    bob_joint_indices[i*3+j] = value*3 + j
            self.bob_joint_indices = bob_joint_indices
        
        self.device=device
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.alice_local_hist_encoder=nn.Linear(alice_joints_num*3,d_model) 

        self.alice_global_hist_encoder=nn.Linear(alice_joints_num*3,d_model)
        self.bob_global_hist_encoder=nn.Linear(bob_joints_num*3,d_model) 

        self.bob_global_future_encoder=nn.Linear(bob_joints_num*3,d_model) 
        
        self.decoder_linear = nn.Linear(d_model,d_model)
        self.forecast_head=nn.Linear(d_model,alice_joints_num*3)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
    
    def forward(self, 
            alice_hist, 
            bob_hist, 
            bob_future,
            add_spe=True, 
            bob_is_robot=False):
        ### This should be zero
        alice_current_pos = alice_hist[:, -1, :].unsqueeze(1)
        
        ### Index out bob's relevant joints
        if self.bob_joint_indices is not None:
            bob_future = bob_future[:,:,self.bob_joint_indices]
            bob_hist = bob_hist[:,:,self.bob_joint_indices]
        
        ### local history encoding
        alice_displacement = alice_hist[:,1:alice_hist.shape[1],:]-alice_hist[:,:alice_hist.shape[1]-1,:]                 
        alice_displacement_dct = dct.dct(alice_displacement)
        alice_local_enc = self.alice_local_hist_encoder(alice_displacement_dct)
        alice_local_output, *_ = self.encoder(alice_local_enc, 1, None)

        ### global history encoding
        alice_global_enc = self.alice_global_hist_encoder(alice_hist)
        if not self.one_hist:
            bob_global_enc = self.bob_global_hist_encoder(bob_hist)
            global_enc = torch.cat([alice_global_enc, bob_global_enc],dim=1)
        else:
            global_enc = alice_global_enc
        global_output, *_ = self.encoder_global(global_enc,
                    1 if self.one_hist else 2, 
                    src_mask = None, 
                    global_feature=True)

        ### conditional future encoder
        if not self.conditional_forecaster:
            bob_future.fill_(0)
        bob_cond_future_enc = self.bob_global_future_encoder(bob_future)

        spe = 0
        if add_spe:
            alice_spe = torch.norm(alice_hist-alice_hist[:, -1].unsqueeze(1), dim=-1)
            bob_spe = torch.norm(bob_hist-alice_hist[:, -1, self.bob_joint_indices].unsqueeze(1), dim=-1)
            spe = torch.exp(-torch.cat([alice_spe, bob_spe] if not self.one_hist else [alice_spe], dim=1)).unsqueeze(2)

        encoder_output = torch.cat([alice_local_output, global_output+spe], dim=1)

        dec_output, dec_attention, *_ = self.decoder(bob_cond_future_enc, None, encoder_output, None)
        dec_output = self.decoder_linear(dec_output)
        alice_forecasts_dct = self.forecast_head(dec_output)
        alice_forecasts_displacments = dct.idct(alice_forecasts_dct)

        alice_forecasts = torch.cumsum(alice_forecasts_displacments, dim=1)

        return alice_forecasts
    
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, 
            dropout=0.2, n_position=100, 
            conditional_forecaster=False,
            alice_joints_num = 9,
            bob_joints_num = 9,
            device='cuda'):

        super().__init__()
        
        self.device=device
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj=nn.Linear(JOINT_AMT*3,d_model) # 45: 15jointsx3
        self.proj2=nn.Linear(JOINT_AMT*3,d_model)
        self.cond_future_embed = nn.Linear(JOINT_AMT*3,d_model)
        self.bob_robot_hist_embed = nn.Linear(BOB_ROBOT_JOINT_AMT*3,d_model)
        self.bob_human_hist_embed = nn.Linear(BOB_HUMAN_JOINT_AMT*3,d_model)
        self.proj_inverse=nn.Linear(d_model,JOINT_AMT*3)
        self.cond_decoder_linear = nn.Linear(d_model,d_model)
        self.l1=nn.Linear(d_model, d_model*4)
        self.l2=nn.Linear(d_model*4, d_model*15)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward_local(self, src_seq, trg_seq, input_seq,use=None):
        
        #only use local-range encoder
        
        n_person=input_seq.shape[1]
        
        src_mask = (torch.ones([src_seq.shape[0],1,src_seq.shape[1]])==True).to(self.device)
        

        trg_mask = (torch.ones([trg_seq.shape[0],1,trg_seq.shape[1]])==True).to(self.device) & get_subsequent_mask(trg_seq).to(self.device)
        

        src_seq_=self.proj(src_seq)
        trg_seq_=self.proj2(trg_seq)
        
        enc_output, *_=self.encoder(src_seq_,n_person, src_mask)
        dec_output, *_=self.decoder(trg_seq_, None, enc_output, None)

        dec_output=self.l1(dec_output)
        dec_output=self.l2(dec_output)
        dec_output=dec_output.view(dec_output.shape[0],-1,self.d_model)        

        dec_output=self.proj_inverse(dec_output)
              
        return dec_output

    def forward(self, src_seq, trg_seq, alice_hist, bob_hist, common_joints, use=None, cond_future=None, bob_is_robot=False, one_hist=False):
        '''
        Produces one forecast
        src_seq: local
        trg_seq: local
        input_seq: global
        cond_future: future plan of the other agent that is being conditioned
        '''
        # n_person=input_seq.shape[1]
        #src_mask = (torch.ones([src_seq.shape[0],1,src_seq.shape[1]])==True).to(self.device)
        # import pdb; pdb.set_trace()
        src_seq_=self.proj(src_seq)
        trg_seq_=self.proj2(trg_seq)
        trg_seq_ = torch.cat([trg_seq_, self.cond_future_embed(cond_future)],dim=1)

        enc_output, *_ = self.encoder(src_seq_, 1, None)
        
        # TODO: Dimensions of Alice's joints won't match Bob's anymore, so use a different
        # MLP to get Bob's joint history into the same temp embedding dimension and concat
        # Note: there will be two different MLPs depending on if Bob is a human vs. robot
        # others=input_seq[:,:,:,:].view(input_seq.shape[0],-1,JOINT_AMT*3)
        others_=self.proj2(alice_hist)
        others_ = others_.unsqueeze(1)
        if bob_is_robot:
            bob_embed = self.bob_robot_hist_embed(bob_hist)
        else:
            bob_embed = self.bob_human_hist_embed(bob_hist)
        if not one_hist:
            others_ = torch.cat([others_, bob_embed.unsqueeze(1)],dim=1)
        others_ = others_[:,:,:,:].view(others_.shape[0],-1,self.d_model)
        mask_other=None
        mask_dec=None

        #mask_other=torch.zeros([others.shape[0],1,others_.shape[1]]).to(self.device).long()
        #for i in range(len(use)):
        #    mask_other[i][0][:use[i]*15]=1
        enc_others,*_=self.encoder_global(others_,1 if one_hist else 2, mask_other, global_feature=True)
        # enc_others=enc_others.unsqueeze(1).expand(input_seq.shape[0],input_seq.shape[1],-1,self.d_model)
        
        enc_others=enc_others.reshape(enc_output.shape[0],-1,self.d_model)
        #mask_other=mask_other.unsqueeze(1).expand(input_seq.shape[0],input_seq.shape[1],1,-1)
        #mask_other=mask_other.reshape(enc_others.shape[0],1,-1)
        #mask_dec=torch.cat([src_mask*1,mask_other.long()],dim=-1)

        # temp_a=input_seq.unsqueeze(1).repeat(1,input_seq.shape[1],1,1,1)
        # temp_b=input_seq[:,:,-1:,:].unsqueeze(2).repeat(1,1,input_seq.shape[1],1,1)
        common_joint_hists = alice_hist[:,:,common_joints].unsqueeze(1)
        if not one_hist:
            common_joint_hists = torch.cat([common_joint_hists, bob_hist[:,:,common_joints].unsqueeze(1)],dim=1)
        temp_a = common_joint_hists
        temp_b = common_joint_hists[:,:,-1:,:]
        
        c=torch.mean((temp_a-temp_b)**2,dim=-1)
        
        # c=c.reshape(c.shape[0]*c.shape[1],c.shape[2]*c.shape[3],1)
        c=c.reshape(c.shape[0],c.shape[1]*c.shape[2],1)
        
        enc_output=torch.cat([enc_output,enc_others+torch.exp(-c)],dim=1)
        dec_output, dec_attention,*_ = self.decoder(trg_seq_[:,:,:], None, enc_output, mask_dec)
        # if self.cond_decoder_linear:
        dec_output = self.cond_decoder_linear(dec_output)
        dec_output=dec_output[:,:15]
        # else:
        #     dec_output= self.l1(dec_output)
        #     dec_output= self.l2(dec_output)
        #     dec_output=dec_output.view(dec_output.shape[0],15,self.d_model)
        
        dec_output=self.proj_inverse(dec_output)
        
        return dec_output#,dec_attention


def forward_multi_predict(self, src_seq, trg_seq, input_seq, use=None):
        '''
        Allows for producing forecasts of multiple humans at once
        src_seq: local
        trg_seq: local
        input_seq: global
        '''
        n_person=input_seq.shape[1]

        #src_mask = (torch.ones([src_seq.shape[0],1,src_seq.shape[1]])==True).to(self.device)
        # import pdb; pdb.set_trace()
        src_seq_=self.proj(src_seq)
        # import pdb; pdb.set_trace()
        trg_seq_=self.proj2(trg_seq)

        enc_output, *_ = self.encoder(src_seq_, n_person, None)
        
        others=input_seq[:,:,:,:].view(input_seq.shape[0],-1,JOINT_AMT*3)
        others_=self.proj2(others)
        mask_other=None
        mask_dec=None

        #mask_other=torch.zeros([others.shape[0],1,others_.shape[1]]).to(self.device).long()
        #for i in range(len(use)):
        #    mask_other[i][0][:use[i]*15]=1

        enc_others,*_=self.encoder_global(others_,n_person, mask_other, global_feature=True)
        enc_others=enc_others.unsqueeze(1).expand(input_seq.shape[0],input_seq.shape[1],-1,self.d_model)
        
        enc_others=enc_others.reshape(enc_output.shape[0],-1,self.d_model)
        #mask_other=mask_other.unsqueeze(1).expand(input_seq.shape[0],input_seq.shape[1],1,-1)
        #mask_other=mask_other.reshape(enc_others.shape[0],1,-1)
        #mask_dec=torch.cat([src_mask*1,mask_other.long()],dim=-1)
        

        temp_a=input_seq.unsqueeze(1).repeat(1,input_seq.shape[1],1,1,1)
        temp_b=input_seq[:,:,-1:,:].unsqueeze(2).repeat(1,1,input_seq.shape[1],1,1)
        
        c=torch.mean((temp_a-temp_b)**2,dim=-1)
        
        c=c.reshape(c.shape[0]*c.shape[1],c.shape[2]*c.shape[3],1)
        # import pdb; pdb.set_trace()
        
        enc_output=torch.cat([enc_output,enc_others+torch.exp(-c)],dim=1)
        dec_output, dec_attention,*_ = self.decoder(trg_seq_[:,:1,:], None, enc_output, mask_dec)
        

        # import pdb; pdb.set_trace()
        dec_output= self.l1(dec_output)
        dec_output= self.l2(dec_output)
        dec_output=dec_output.view(dec_output.shape[0],15,self.d_model)
        
        dec_output=self.proj_inverse(dec_output)
        
        return dec_output#,dec_attention



class Discriminator(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2, n_position=50,
            device='cuda'):

        super().__init__()
        self.device=device     
        self.d_model=d_model
        self.encoder = Encoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout, device=self.device)
            
        self.fc=nn.Linear(JOINT_AMT*3,1)
        
    
    def forward(self, x):
        y, *_ = self.encoder(x,n_person=None, src_mask=None)
        y=self.fc(y)
        y=y.view(-1,1)
        return y