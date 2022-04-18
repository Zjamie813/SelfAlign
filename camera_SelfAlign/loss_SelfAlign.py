import torch.nn.functional as F
import torch
import torch.nn as nn

class LCA_loss(nn.Module):
    def __init__(self,in_dim, nmb_prototypes=4096,student_temp=0.1, epsilon=0.05, sinkhorn_iterations=3):
        super(LCA_loss, self).__init__()
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.in_dim = in_dim
        self.out_dim = nmb_prototypes
        self.student_temp = student_temp
        self.v_prototypes = nn.Linear(in_dim, nmb_prototypes, bias=False)

    def distributed_sinkhorn(self, out, epsilon=0.05, sinkhorn_iterations=3):
        Q = torch.exp(out / epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)  # [1]
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)  # [3000,1]
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, v_feats, t_feats, t_lengths):
        '''
        :param v_feats:  the target feature, i.e. visual concepts, [bs,num_region,dim]
        :param t_feats:  the source feature, i.e. textual concepts, [bs, num_word, dim]
        :param t_lengths:  meaningful text concept length each sentence, [bs]
        '''
        # normalize the prototypes
        with torch.no_grad():
            v_w = self.v_prototypes.weight.data.clone()  # [potypes,dim]  here is 4096 * 2048
            v_w = nn.functional.normalize(v_w, dim=1, p=2)
            self.v_prototypes.weight.copy_(v_w)

        bt, v_len, emb_size = v_feats.size()
        max_len = t_feats.size(1)

        v_proj_out = self.v_prototypes(v_feats.contiguous().view(bt * v_len, -1))
        t_proj = self.v_prototypes(t_feats.contiguous().view(bt * max_len, -1))
        t_proj = t_proj / self.student_temp

        # online clustering
        with torch.no_grad():
            v_out = v_proj_out.detach()
            v_probs = self.distributed_sinkhorn(v_out, epsilon=self.epsilon,sinkhorn_iterations=self.sinkhorn_iterations)

        v_probs = v_probs.view(bt, v_len, -1)
        t_proj = t_proj.view(bt, max_len, -1)

        total_loss = 0

        for i in range(bt):
            n_word = t_lengths[i]
            v_region_fea, v_region_prob = v_feats[i], v_probs[i]  # [len,dim]
            t_region_fea, t_region_prob = t_feats[i][:n_word, :], t_proj[i][:n_word, :]

            # similarity matrix between region features, row denotes text,column denotes image
            region_sim_matrix = torch.matmul(t_region_fea, v_region_fea.permute(1, 0))  # [t_len,v_len]
            region_sim_idx = region_sim_matrix.argmax(dim=1)  # [t_len]
            expand_region_sim_idx = region_sim_idx.unsqueeze(1).expand(-1, self.out_dim)
            v_indexed_prob = v_region_prob.gather(0, expand_region_sim_idx)  # [n_word,65536]

            loss_gird = torch.sum(-v_indexed_prob * F.log_softmax(t_region_prob, dim=-1), dim=[-1]).mean(-1)

            total_loss = total_loss + loss_gird

        local_loss = total_loss / bt

        return local_loss

def CRA_loss(l, m, T=1, neg_num=None):
    '''Computes the noise contrastive estimation-based loss
    Args:
        l: [B,dim,n_l],keys
        m: [B,dim,n_g],query
        neg_num: the number of negatives from other pair
        neg_mask: if
    Returns:
        torch.Tensor: Loss.
    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1) # [bt,n_loclas,dim]
    m_p = m.permute(0, 2, 1) # [bt,n_multis,dim]

    l_n = l_p.reshape(-1, units) # [bt*n_locals,dim]
    m_n = m_p.reshape(-1, units) # [bt*1,dim]

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2) # [B,n_locals,dim] * [B,dim,1] ->[bt,n_locals,1,1]
    u_n = torch.mm(m_n, l_n.t()) # [B*1,dim] * [dim,B*36] ->[bt*1,b*n_locals]

    # add apply tao
    u_p = u_p / T
    u_n = u_n / T

    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1) # N,N,n_locals,n_multis

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device) # [bt,bt,1,1]
    n_mask = 1 - mask
    n_mask = n_mask.expand(-1,-1,n_locals,n_multis)

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)

    # hard negatives
    if neg_num:
        sort_u_n,_ = torch.topk(u_n,dim=2,k=neg_num)
        pred_lgt = torch.cat([u_p,sort_u_n],dim=2)

    pred_log = F.log_softmax(pred_lgt, dim=2) # [bt,n_locals,neg_num+1,n_multis]


    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss
