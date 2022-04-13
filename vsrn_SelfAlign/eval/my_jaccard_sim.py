# -*- coding: utf-8 -*-
import numpy
import torch
def jaccard_sim_in_HNET(im, s):
    im_bs = im.size(0)
    s_bs = s.size(0)
    im = im.unsqueeze(1).expand(-1, s_bs, -1)
    s = s.unsqueeze(0).expand(im_bs, -1, -1)
    intersection = numpy.min(im, s).sum(-1)
    union = numpy.max(im, s).sum(-1)
    score = intersection / union
    return score

def jaccard_sim_num(im,s):
    im_bs = im.shape[0]
    s_bs = s.shape[0]
    im = numpy.expand_dims(im,axis=1).repeat(s_bs,1)
    s = numpy.expand_dims(s, axis=0).repeat(im_bs, 0)
    intersection = numpy.minimum(im, s).sum(-1)
    union = numpy.maximum(im, s).sum(-1)
    score = intersection / union
    return score

def tensor_get_cls_label(images):# 获取整体的类标，如果存在这个类就为1，不存在为0
    # images:[bt,loc_num,n_cls]
    # return [bt,n_cls]
    n_cls = images.size(2)
    max_v_cls = torch.max(images, dim=2)[0]

    max_v_cls = max_v_cls.unsqueeze(-1).expand(-1,-1,n_cls)
    oh_v = torch.ge(images, max_v_cls)  # >=
    v_cls = torch.sum(oh_v, dim=1)  # [bt,4096],多分类

    # 多个类别出现即为1，不记出现次数
    v_one = torch.ones_like(v_cls)
    v_zero = torch.zeros_like(v_cls)
    v_cls = torch.where(v_cls > 0.5, v_one, v_zero)  # [bt,n_cls]

    return v_cls

def get_cls_label(images):# 获取整体的类标，如果存在这个类就为1，不存在为0
    # images:[bt,loc_num,n_cls]
    # return [bt,n_cls]
    n_cls = images.shape[2]
    max_v_cls = numpy.max(images, axis=2)

    max_v_cls = numpy.expand_dims(max_v_cls,axis=-1).repeat(n_cls,-1)
    oh_v = numpy.greater_equal(images, max_v_cls)  # >=
    v_cls = numpy.sum(oh_v, axis=1)  # [bt,4096],多分类

    # 多个类别出现即为1，不记出现次数
    v_one = numpy.ones_like(v_cls)
    v_zero = numpy.zeros_like(v_cls)
    v_cls = numpy.where(v_cls > 0.5, v_one, v_zero)  # [bt,n_cls]

    return v_cls

def my_jaccard_sim(captions,images):
    # images:[bt,36,4096],captions:[bt,max_len,4096]
    t_cls = get_cls_label(captions).astype(numpy.float)  # 变成了one hot向量，是不是可以考虑类别次数？
    v_cls = get_cls_label(images).astype(numpy.float)

    print(t_cls)
    print(v_cls)
    score = j_sim(t_cls,v_cls) # t is the query
    return score # [c_bs,im_bs]

def j_sim(c,im):
    # [bt,dim]
    im_bs = im.shape[0]
    c_bs = c.shape[0]
    im = numpy.expand_dims(im,0).repeat(c_bs,0)
    c = numpy.expand_dims(c,1).repeat(im_bs,1)
    fenmu = numpy.sum(c,axis=-1)
    fenzi = numpy.minimum(c,im).sum(axis=-1) # 这里的min是从两个向量中取小的那方，相当于取交集操作
    score = fenzi / fenmu
    return score


a = torch.rand(2,3,4).cuda()
print(a)
print(tensor_get_cls_label(a))
# b = numpy.random.rand(2,2,4)
# print(a)
# print(b)
# print(my_jaccard_sim(a,b))
# a = numpy.array([[0, 2, 0, 1],[1, 1, 1, 0]]).astype(numpy.float)
# b = numpy.array([[0, 1, 0, 1],[0, 2, 0, 0]]).astype(numpy.float)
# s = j_sim(b,a)
# print(s)

