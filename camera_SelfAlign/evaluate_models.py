import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
reload(sys)
sys.setdefaultencoding('utf-8')

import evaluation
# torch.backends.cudnn.enabled=False


# flickr
# evaluation_models.evalrank_single(Trained_model1_path, data_path=DATA_PATH, split="test", fold5=False)
# evaluation_models.evalrank_ensemble(Trained_model1_path, Trained_model2_path, \
#                     data_path=DATA_PATH, split="test", fold5=False)

# coco 5k
# evaluation_models.evalrank_single(Trained_model1_path, data_path=DATA_PATH, split="testall", fold5=True)
# evaluation_models.evalrank_ensemble(Trained_model1_path, Trained_model2_path, \
#                     data_path=DATA_PATH, split="testall", fold5=False)

# coco 1k
# evaluation_models.evalrank_single(Trained_model1_path, data_path=DATA_PATH, split="testall", fold5=True)
# evaluation_models.evalrank_ensemble(Trained_model1_path, Trained_model2_path, \
#                     data_path=DATA_PATH, split="testall", fold5=True)
