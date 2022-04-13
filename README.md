# SelfAlign: A Self-Supervised Align Module for Fast and Accurate Image-Text Matching
Pytorch code of the paper "SelfAlign: Towards Fast and Accurate Image-Text Retrieval with Self-Supervised Fine-Grained Alignment".  It is built on top of VSRN and CAMERA.

# Introduction

Image-text retrieval requires the system to bridge the heterogenous gap between vision and language for accurate retrieval while keeping the network lightweight-enough for efficient retrieval. Existing trade-off solutions mainly study from the view of incorporating cross-modal interactions with the independent-embedding framework or leveraging stronger pre-trained encoders, which still demand  time-consuming similarity measurement or heavyweight model structure in the retrieval stage. In this work, we propose a image-text alignment module SelfAlign on top of independent-embedding framework, which improves the retrieval accuracy  while maintaining the retrieval efficiency without extra supervision. 

SelfAlign contains two collaborative sub-modules that force image-text alignment at both concept level and context level by self-supervised contrastive learning, which doesn’t require cross-modal embedding interactions in training and maintains independent image and text encoders in retrieval. 

With comparable time cost, SelfAlign consistently boosts the accuracy of state-of-the-art independent-embedding models respectively by 9.1\%, 4.2\% and 6.6\% on Flickr30K, MSCOCO1K and MSCOCO5K. The retrieval accuracy also outperforms most of existing interactive-embedding models with  orders of magnitude decrease in retrieval time.


## Requirements   
We recommended the following dependencies.  
  
* Python 2.7   
* [PyTorch](http://pytorch.org/) (0.4.1)  
* [Transformers]() 2.1.1
* [NumPy](http://www.numpy.org/) (>1.12.1)  
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)  
* [pycocotools](https://github.com/cocodataset/cocoapi)  
* [torchvision]()  
* [matplotlib]()  
  
  
* Punkt Sentence Tokenizer:  
```python  
import nltk  
nltk.download()  
> d punkt  
```  
## Download data
Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/).

We follow [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention) and [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features for fair comparison. More details about data pre-processing (optional) can be found [here](https://github.com/kuanghuei/SCAN/blob/master/README.md#data-pre-processing-optional). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:

```bash

wget https://scanproject.blob.core.windows.net/scan-data/data.zip

```

You can also get the data from google drive: https://drive.google.com/drive/u/1/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC. We refer to the path of extracted files for `data.zip` as `$DATA_PATH`.


##  CAMERA + SelfAlign module
### BERT model  
 We use the BERT code from [BERT-pytorch](https://github.com/huggingface/pytorch-transformers). Please following [here](https://github.com/huggingface/pytorch-transformers/blob/4fc9f9ef54e2ab250042c55b55a2e3c097858cb7/docs/source/converting_tensorflow_models.rst) to convert the Google BERT model to a PyTorch save file `$BERT_PATH`.

###  Training new models 
Go to the directory `./camera_SelfAlign`, Run `train_SelfAlign.py`:

For Flickr30K:

```bash

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --logger_name runs/flickr --data_name f30k_precomp --num_epochs 30 --lr_update 10

```

For MSCOCO:

```bash

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --logger_name runs/coco --data_name coco_precomp --num_epochs 40 --lr_update 20

```
###  Evaluate trained models  
  
Modify the model_path and data_path in the `evaluation_models.py` file. Then Run it : 
  
```bash  
python evaluate_models.py  
```  

##  VSRN + SelfAlign module
###  Training new models 
Go to the directory `./vsrn_SelfAlign`, Run `train_SelfAlign.py`:

For Flickr30K:

```bash

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --logger_name runs/flickr --data_name f30k_precomp --lr_update 10

```

For MSCOCO:

```bash

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --logger_name runs/coco --data_name coco_precomp --lr_update 15

```
###  Evaluate trained models  
  
Modify the model_path and data_path in the `evaluation_models.py` file. Then Run it : 
  
```bash  
python evaluate_models.py  
```  

##  License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
