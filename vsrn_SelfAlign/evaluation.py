from __future__ import print_function
import os, sys
import pickle
import argparse

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from tqdm import tqdm

from model_SelfAlign import VSRN
from evaluation_SelfAlign import AverageMeter, LogCollector, encode_data, calItr,i2t_cmr,t2i_cmr
from collections import OrderedDict
from misc.utils import print_options

import time

def evalrank_single(model_path, data_path=None, split='test', fold5=False):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    if data_path is not None:
        opt.data_path = data_path

    model = VSRN(opt)

    ckpt_model = checkpoint['model']
    # load model state
    model.load_state_dict(ckpt_model)

    total_trainable_parameters = sum(p.numel() for p in model.params if p.requires_grad)
    print(total_trainable_parameters)
    # print('Number of parameter: %.2fM'.format(total_trainable_parameters/1e6))

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)
    print('Computing results...')
    enc_sta = time.time()
    imgs, caps = encode_data(model, data_loader)
    enc_end = time.time()
    print('camera encoding time:%.2fs' % (enc_end - enc_sta))
    print('#Images: %d, #Captions: %d' % (imgs.shape[0] / 5, caps.shape[0]))

    if not fold5:
        imgs = numpy.array([imgs[i] for i in range(0, len(imgs), 5)])
        re_start = time.time()
        sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5)
        # no cross-validation, full evaluation
        re_end = time.time()
        print('camera f30k 1K images query retrieval time:%.2fs' % (re_end - re_start))
        r, rt = i2t_cmr(sims, return_ranks=True)
        ri, rti = t2i_cmr(sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            imgs_block, caps_block = imgs[i * 5000:(i + 1) * 5000], caps[i * 5000:(i + 1) * 5000]
            imgs_block = numpy.array([imgs_block[i] for i in range(0, len(imgs_block), 5)])
            sims = calItr(model, imgs_block, caps_block, shard_size=opt.batch_size * 5)
            r, rt0 = i2t_cmr(sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i_cmr(sims, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])


def evalrank_ensemble(model_path, model_path2, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    # Load Vocabulary Wrapper
    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']
    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)

    if data_path is not None:
        opt.data_path = data_path

    model = VSRN(opt)
    model2 = VSRN(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    # (split_name, data_name, batch_size, workers, opt):
    batch_size = 32
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  batch_size, opt.workers, opt)

    print('Computing results...')
    t0 = time.time()

    imgs, caps = encode_data(model, data_loader)
    imgs2, caps2 = encode_data(model2, data_loader)
    if not fold5:
        # no cross-validation, full evaluation
        imgs = numpy.array([imgs[i] for i in range(0, len(imgs), 5)])
        sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5)
        imgs2 = numpy.array([imgs2[i] for i in range(0, len(imgs2), 5)])
        sims2 = calItr(model2, imgs2, caps2, shard_size=opt2.batch_size * 5)
        sims = (sims + sims2) / 2
        r, rt = i2t_cmr(sims, return_ranks=True)
        ri, rti = t2i_cmr(sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            imgs_block, caps_block = imgs[i * 5000:(i + 1) * 5000], caps[i * 5000:(i + 1) * 5000]
            imgs_block = numpy.array([imgs_block[j] for j in range(0, len(imgs_block), 5)])
            sims = calItr(model, imgs_block, caps_block, shard_size=opt.batch_size * 5)
            imgs2_block, caps2_block = imgs2[i * 5000:(i + 1) * 5000], caps2[i * 5000:(i + 1) * 5000]
            imgs2_block = numpy.array([imgs2_block[j] for j in range(0, len(imgs2_block), 5)])
            sims2 = calItr(model2, imgs2_block, caps2_block, shard_size=opt2.batch_size * 5)
            sims = (sims + sims2) / 2

            r, rt0 = i2t_cmr(sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i_cmr(sims, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

