import model
import vpdataset
import train
import torchtext.data as data
import torch.autograd as autograd
import torch
import argparse
import datetime
import os
import glob

from collections import namedtuple

def vp(text_field, label_field, foldid, path=None, filename=None, num_experts=0, **kargs):
    # print('num_experts', num_experts)
    train_data, dev_data, test_data = vpdataset.VP.splits(text_field, label_field, root=path, filename=filename, foldid=foldid,
                                                          num_experts=num_experts)
    if num_experts > 0:
        text_field.build_vocab(train_data[0], dev_data[0], test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                              wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    else:
        text_field.build_vocab(train_data, dev_data, test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    # label_field.build_vocab(train_data, dev_data, test_data)
    kargs.pop('wv_type')
    kargs.pop('wv_dim')
    kargs.pop('wv_dir')
    kargs.pop("min_freq")
    # print(type(train_data), type(dev_data))
    if num_experts > 0:
        train_iter = []
        dev_iter = []
        for i in range(num_experts):
            this_train_iter, this_dev_iter, test_iter = data.Iterator.splits((train_data[i], dev_data[i], test_data),
                                                                             batch_sizes=(50,
                                                                                          len(dev_data[i]),
                                                                                          len(test_data)), **kargs)
            train_iter.append(this_train_iter)
            dev_iter.append(this_dev_iter)
    else:
        train_iter, dev_iter, test_iter = data.Iterator.splits(
            (train_data, dev_data, test_data),
            batch_sizes=(args.batch_size,
                         len(dev_data),
                         len(test_data)),
            **kargs)
    return train_iter, dev_iter, test_iter


parser = argparse.ArgumentParser(description='CNN text classifier')
parser.add_argument('-log-file', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + 'result.txt',
                    help='the name of the file to store results')
parser.add_argument('-model', type=str, default=None, help='path of model snapshot directory [default: None]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-yes-cuda', action='store_true', default=True, help='disable the gpu')

args = parser.parse_args()
log_file_handle = open(args.log_file, 'w')

phn_test_file = 'spoken_test.phone.tsv'

CNN_Args = namedtuple('CNN_Args', ['embed_num',
                                   'char_embed_dim',
                                   'word_embed_dim',
                                   'class_num',
                                   'kernel_num',
                                   'char_kernel_sizes',
                                   'word_kernel_sizes',
                                   'ortho_init',
                                   'dropout',
                                   'static',
                                   'word_vector',
                                   'cuda'])
data_dir = './data/'
word_file = 'wilkins.word.shuffled.30.txt'
phn_file = 'wilkins.phone.shuffled.30.txt'

#Predict_Args = namedtuple('Predict_Args', ['ensemble', 'cuda'])
                                           
#word_tokenizer = data.Pipeline(vpcnn.vpdataset.clean_str)
phn_field = data.Field(lower=True)
#word_field = data.Field(lower=True, tokenize=word_tokenizer, batch_first=True)
label_field = data.Field(sequential=False, use_vocab=False, preprocessing=int)

phn_test_data = vpdataset.VP(phn_field,label_field,path=data_dir, filename=phn_test_file)
phn_test_iter = data.Iterator(phn_test_data, len(phn_test_data))
train_iter, dev_iter, test_iter = vp(phn_field, 
                                     label_field, 
                                     path=data_dir, 
                                     filename=phn_file, 
                                     foldid=0, 
                                     num_experts=5,
                                     repeat=False, 
                                     sort=False, 
                                     wv_type=None, 
                                     wv_dim=None, 
                                     wv_dir=None, 
                                     min_freq=1)


#phn_field.build_vocab(phn_test_data,
#                      wv_type=None,
#                      wv_dim=None,
#                      wv_dir=None,
#                      min_freq=1)
#word_train_data, word_dev_data, word_test_data = vpcnn.vpdataset.VP.splits(word_field,
#                                                                           label_field,
#                                                                           foldid = 1,
#                                                                           num_experts = 5)
#word_field.build_vocab(word_train_data[0],
#                       word_dev_data[0],
#                       word_test_data,
#                       wv_type=None, 
#                       wv_dim=None, 
#                       wv_dir=None, 
#                       min_freq=1)
print("phn len", len(phn_field.vocab))
phn_args = CNN_Args(embed_num = len(phn_field.vocab),
                    char_embed_dim = 16,
                    word_embed_dim = 300,
                    class_num = 359,
                    kernel_num = 400,
                    char_kernel_sizes = [2,3,4,5,6],
                    word_kernel_sizes = [3,4,5],
                    ortho_init = False,
                    dropout = 0.5,
                    static = args.static,
                    word_vector = 'w2v',
                    cuda = args.yes_cuda and torch.cuda.is_available())  # ; del args.no_cuda

phn_mdl_path = os.path.join(args.model, '*')
phn_mdl_files = glob.glob(phn_mdl_path)
phn_mdls = []
for i in range(len(phn_mdl_files)):
    phn_mdls.append(model.CNN_Text(phn_args, 'char'))
    phn_mdls[i].load_state_dict(torch.load(phn_mdl_files[i], map_location= lambda stor, loc: stor))
#word_args = CNN_Args(embed_num = len(word_field.vocab), ## (should be 1715)
#                     char_embed_dim = 16,
#                     word_embed_dim = 300,
#                     class_num = 359,
#                     kernel_num = 300,
#                     char_kernel_sizes = [2,3,4,5,6],
#                     word_kernel_sizes = [3,4,5],
#                     ortho_init = False,
#                     dropout = 0.5,
#                     static = False,
#                     word_vector = 'w2v')
#word_mdl_path = os.path.join(conf['word_cnn_dir'], '*')
#word_mdl_files = glob.glob(word_mdl_path)
#word_mdls = []
#for i in range(len(word_mdl_files)):
#    word_mdls.append(vpcnn.model.CNN_Text(word_args, 'word'))
#    word_mdls[i].load_state_dict(torch.load(word_mdl_files[i], map_location= lambda stor, loc: stor))

result = train.ensemble_eval(phn_test_iter, phn_mdls, phn_args, log_file_handle=log_file_handle)
print("Accuracy on Test: {1} for PHN".format(result))
print("Accuracy on Test: {1} for PHN".format(result), file=log_file_handle)
