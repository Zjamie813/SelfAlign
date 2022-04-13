import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

from models import PositionEncoder,AGSA,Summarization,MultiViewMatching
from loss import TripletLoss, DiversityRegularization
from transformers import  BertModel, BertConfig
from loss_SelfAlign import LCA_loss, CRA_loss

def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def init_fc(fc):
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)
    return fc

class EncoderImagePrecompSelfAttn(nn.Module):

    def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0):
        super(EncoderImagePrecompSelfAttn, self).__init__()
        self.embed_size = embed_size

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        self.position_enc = PositionEncoder(embed_size)
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        self.mvs = Summarization(embed_size, smry_k)

        # CRA
        self.ctx_fc = nn.Linear(embed_size, embed_size)
        self.ctx_fc_bn1 = nn.BatchNorm1d(embed_size)
        self.ctx_fc_bn2 = nn.BatchNorm1d(embed_size)
        self.fusion_fn = nn.Sequential(
            nn.Linear(embed_size + embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, boxes, imgs_wh,ctx_fc_relu):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(fc_img_emd)  #(bs, num_regions, dim)
        posi_emb = self.position_enc(boxes, imgs_wh)    #(bs, num_regions, num_regions, dim)

        # Adaptive Gating Self-Attention
        self_att_emb1 = self.agsa(fc_img_emd, posi_emb)    #(bs, num_regions, dim)
        # for baseline model
        self_att_emb = l2norm(self_att_emb1)
        # global
        img_avg = torch.mean(self_att_emb1, dim=1)
        # img_glo_guide = self.ctx_fc(img_avg)
        bs,num_regions,dim = self_att_emb1.size()
        img_glo_guide = self.ctx_fc_bn1(self.ctx_fc(img_avg))
        self_att_emb2 = self.ctx_fc_bn2(self_att_emb1.view(-1,dim)).view(bs,num_regions,dim)
        if ctx_fc_relu:
            img_glo_guide = F.relu(img_glo_guide)
            self_att_emb2 = F.relu(self_att_emb2)

        # Multi-View Summarization
        smry_mat = self.mvs(self_att_emb)
        L = F.softmax(smry_mat, dim=1)
        img_emb_mat = torch.matmul(L.transpose(1, 2), self_att_emb) #(bs, k, dim)

        # CRA fusion
        m = torch.cat([img_avg, img_glo_guide], dim=-1)
        g = torch.sigmoid(self.fusion_fn(m)).expand(-1, self.embed_size)  # [bt,1]
        _g = (1 - g).expand(-1, self.embed_size)
        img_ctx_fea = g * img_avg + _g * img_glo_guide

        return F.normalize(img_emb_mat, dim=-1), smry_mat, F.normalize(fc_img_emd,p=2,dim=-1), F.normalize(img_glo_guide,p=2,dim=-1),F.normalize(self_att_emb2,p=2,dim=-1),F.normalize(img_ctx_fea,p=2,dim=-1), fc_img_emd, self_att_emb

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

class TextEncoder(nn.Module):
    """
    """
    def __init__(self,opt, cfg_file, init_ckpt, embed_size, head, drop=0.0):
        super(TextEncoder, self).__init__()
        self.embed_size = embed_size
        bert_config = BertConfig.from_json_file(cfg_file)
        # bert_config = BertConfig.from_pretrained(config['text-model']['pretrain'],
        #                                          output_hidden_states=True,
        #                                          num_hidden_layers=config['text-model']['extraction-hidden-layer'])
        self.bert = BertModel.from_pretrained(init_ckpt, config=bert_config)
        freeze_layers(self.bert)

        self.mapping = nn.Linear(bert_config.hidden_size, embed_size)
        self.init_weights()
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        # MLP
        hidden_size = embed_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(drop)

        # CRA
        self.ctx_fc = nn.Linear(embed_size, embed_size)
        self.ctx_fc_bn1 = nn.BatchNorm1d(embed_size)
        self.ctx_fc_bn2 = nn.BatchNorm1d(embed_size)
        self.fusion_fn = nn.Sequential(
            nn.Linear(embed_size + embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features +
                                  self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)

    def forward(self, input_ids, attention_mask, token_type_ids, lengths,ctx_fc_relu):
        last_encoder_emb, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        locals = self.mapping(last_encoder_emb)
        locals = l2norm(locals)
        x = locals.clone()
        bs, token_num = x.size()[:2]
        agsa_emb = self.agsa(x)

        # CRA
        glo_fea = torch.mean(agsa_emb, dim=1)
        bs, num_regions, dim = agsa_emb.size()
        atd_glo_fea = self.ctx_fc_bn1(self.ctx_fc(glo_fea))
        agsa_emb1 = self.ctx_fc_bn2(agsa_emb.view(-1, dim)).view(bs, num_regions, dim)
        if ctx_fc_relu:
            atd_glo_fea = F.relu(atd_glo_fea)
            agsa_emb1 = F.relu(agsa_emb1)

        m = torch.cat([glo_fea, atd_glo_fea], dim=-1)
        g = torch.sigmoid(self.fusion_fn(m)).expand(-1, self.embed_size)
        _g = (1 - g).expand(-1, self.embed_size)
        cap_ctx_fea = g * glo_fea + _g * atd_glo_fea

        x = self.fc2(self.dropout(F.relu(self.fc1(agsa_emb))))
        x = (self.bn(x.view(bs*token_num, -1))).view(bs, token_num, -1)
        x = agsa_emb + self.dropout(x)    # context-enhanced word embeddings

        cap_emb = torch.mean(x, 1)
        return F.normalize(cap_emb, p=2, dim=-1), F.normalize(locals, p=2, dim=-1),F.normalize(atd_glo_fea,p=2,dim=-1),F.normalize(agsa_emb1,p=2,dim=-1),F.normalize(cap_ctx_fea,p=2,dim=-1), locals, agsa_emb

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in own_state.items():
            if name in state_dict:
                new_state[name] = state_dict[name]
            else:
                new_state[name] = param

        super(TextEncoder, self).load_state_dict(new_state)

class CAMERA(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.v5_tao = opt.v5_tao
        self.v6_tao = opt.v6_tao
        self.local_clu_num = opt.local_clu_num
        self.ctx_fc_relu = opt.ctx_fc_relu
        self.img_enc = EncoderImagePrecompSelfAttn(opt.img_dim, opt.embed_size, \
                                    opt.head, opt.smry_k, drop=opt.drop)
        self.txt_enc = TextEncoder(opt, opt.bert_config_file, opt.init_checkpoint, \
                                    opt.embed_size, opt.head, drop=opt.drop)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        self.mvm = MultiViewMatching()
        # Loss and Optimizer
        self.region_loss = LCA_loss(in_dim=opt.embed_size, student_temp=0.1, nmb_prototypes=self.local_clu_num, epsilon=0.05, sinkhorn_iterations=2)
        self.crit_ranking = TripletLoss(margin=opt.margin, max_violation=opt.max_violation)
        self.crit_div = DiversityRegularization(opt.smry_k, opt.batch_size)

        self.region_loss.cuda()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.region_loss.parameters())
        params = filter(lambda p: p.requires_grad, params)

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),self.region_loss.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.region_loss.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.region_loss.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.region_loss.eval()

    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, boxes, imgs_wh, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            boxes = boxes.cuda()
            imgs_wh = imgs_wh.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        # Forward
        cap_emb,cap_local_emb, cap_atd_glo, cap_locals, cap_ctx_fea, cap_word_emb, cap_ctx_emb = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths,self.ctx_fc_relu)
        img_emb, smry_mat,img_fc_emb, img_atd_glo,img_locals,img_ctx_fea, img_region_emb, img_ctx_emb = self.img_enc(images, boxes, imgs_wh,self.ctx_fc_relu)

        if volatile:
            img_ctx_fea = img_ctx_fea * 0.5
            img_locals = img_locals * 0.5
        return img_emb, cap_emb, smry_mat,img_fc_emb,cap_local_emb,\
               img_atd_glo,cap_atd_glo, img_locals, cap_locals,img_ctx_fea,cap_ctx_fea


    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        self_att_emb, cap_emb, smry_mat, img_fc_emb, cap_local_emb, \
        img_atd_glo, cap_atd_glo, img_locals, cap_locals, img_ctx_fea_fuse, cap_ctx_fea_fuse = self.forward_emb(batch_data)
        bs, num_view = self_att_emb.size()[:2]
        # local loss
        L_loss = self.region_loss(img_fc_emb, cap_local_emb, t_lengths=batch_data[4])
        self.logger.update('Local', L_loss.item(), bs)

        # contrast in textually-global to visually-local
        cap_atd_glo = cap_atd_glo.unsqueeze(1)
        ctx_t_q = cap_atd_glo.permute(0, 2, 1)
        ctx_v_k = img_locals.permute(0, 2, 1)
        v2t_ctx_loss = CRA_loss(l=ctx_v_k, m=ctx_t_q, neg_num=512, T=self.v5_tao)
        # self.logger.update('v2t_ctx_loss', v2t_ctx_loss.item(), bs)
        img_ctx_v1 = torch.mean(img_locals, dim=1)
        cap_ctx_v1 = cap_ctx_fea_fuse

        # contrast in visually-global to textually-local
        img_atd_glo = img_atd_glo.unsqueeze(1)
        ctx_v_q = img_atd_glo.permute(0, 2, 1)
        ctx_t_k = cap_locals.permute(0, 2, 1)
        t2v_ctx_loss = CRA_loss(l=ctx_t_k, m=ctx_v_q, neg_num=512,T=self.v6_tao)
        # self.logger.update('t2v_ctx_loss', t2v_ctx_loss.item(), bs)

        img_ctx_v2 = img_ctx_fea_fuse
        cap_ctx_v2 = torch.mean(cap_locals, dim=1)

        img_ctx_fea = torch.cat((img_ctx_v1, img_ctx_v2), dim=-1)
        img_ctx_fea = img_ctx_fea * 0.5
        cap_ctx_fea = torch.cat((cap_ctx_v1, cap_ctx_v2), dim=-1)

        ctx_loss = 0.5 * t2v_ctx_loss + 0.5 * v2t_ctx_loss
        self.logger.update('ctx_loss', ctx_loss.item(), bs)

        # bidirectional triplet ranking loss
        self_att_emb = torch.cat((self_att_emb, img_ctx_fea.unsqueeze(1).expand(bs, num_view, -1)), dim=-1)
        cap_emb = torch.cat((cap_emb, cap_ctx_fea), dim=-1)
        sim_mat = self.mvm(self_att_emb, cap_emb)
        ranking_loss = self.crit_ranking(sim_mat)
        self.logger.update('Rank', ranking_loss.item(), bs)
        # diversity regularization
        div_reg = self.crit_div(smry_mat)
        self.logger.update('Div', div_reg.item(), bs)
        # total loss
        loss = ranking_loss + div_reg * self.opt.smry_lamda + L_loss + ctx_loss
        self.logger.update('Le', loss.item(), bs)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            if isinstance(self.params[0], dict):
                params = []
                for p in self.params:
                    params.extend(p['params'])
                clip_grad_norm(params, self.grad_clip)
            else:
                clip_grad_norm(self.params, self.grad_clip)
        if epoch < 1:
            for name, p in self.region_loss.named_parameters():
                if "v_prototypes" in name:
                    p.grad = None

        self.optimizer.step()

