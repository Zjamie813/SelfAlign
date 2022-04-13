import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from bert import BertModel, BertConfig
from transformers import BertModel, BertConfig
from camera_SelfAlign.models import AGSA


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class TextEncoder(nn.Module):
    """
    """
    def __init__(self, cfg_file, init_ckpt, embed_size, head, drop=0.0):
        super(TextEncoder, self).__init__()
        bert_config = BertConfig.from_json_file(cfg_file)
        self.bert = BertModel.from_pretrained(init_ckpt, config=bert_config)
        freeze_layers(self.bert)

        self.mapping = nn.Linear(bert_config.hidden_size, embed_size)
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        # MLP
        hidden_size = embed_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, input_ids, attention_mask, token_type_ids, lengths):
        last_encoder_emb, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        local_emb = self.mapping(last_encoder_emb)    #(bs, token_num, final_dim)

        bs, token_num = local_emb.size()[:2]
        agsa_emb = self.agsa(local_emb)
        x = self.fc2(self.dropout(F.relu(self.fc1(agsa_emb))))
        x = (self.bn(x.view(bs*token_num, -1))).view(bs, token_num, -1)  
        x = agsa_emb + self.dropout(x)    # context-enhanced word embeddings

        cap_emb = torch.mean(x, 1) #
        return F.normalize(cap_emb, p=2, dim=-1)

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
