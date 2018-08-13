import vpcnn.model
import vpcnn.vpdataset 
import vpcnn.train
import torchtext.data as data
import torch.autograd as autograd
import torch
import argparse

from collections import namedtuple
parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-log-file', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + 'result.txt',
                    help='the name of the file to store results')
parser.add_argument('-model', type=str, default=None, help='path of model snapshot directory [default: None]')

args = parser.parse_args()
log_file_handle = open(args.log_file, 'w')

phn_test_file = 'data/spoken_test.phone'

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
                                   'word_vector'])

#Predict_Args = namedtuple('Predict_Args', ['ensemble', 'cuda'])
                                           
#word_tokenizer = data.Pipeline(vpcnn.vpdataset.clean_str)
phn_field = data.Field(lower=True, tokenize=lambda x: x.split())
#word_field = data.Field(lower=True, tokenize=word_tokenizer, batch_first=True)


phn_test_data = vpcnn.vpdataset.VP(phn_field,label_field,path=phn_test_file)
phn_field.build_vocab(phn_test_data,
                      wv_type=None,
                      wv_dim=None,
                      wv_dir=None,
                      min_freq=1)
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
phn_args = CNN_Args(embed_num = len(phn_field.vocab),
                     char_embed_dim = 16,
                     word_embed_dim = 300,
                     class_num = 359,
                     kernel_num = 400,
                     char_kernel_sizes = [2,3,4,5,6],
                     word_kernel_sizes = [3,4,5],
                     ortho_init = False,
                     dropout = 0.5,
                     static = False,
                     word_vector = 'w2v')
phn_mdl_path = os.path.join(args.model, '*')
phn_mdl_files = glob.glob(phn_mdl_path)
phn_mdls = []
for i in range(len(phn_mdl_files)):
    phn_mdls.append(vpcnn.model.CNN_Text(phn_args, 'char'))
    phn_mdls[i].load_state_dict(torch.load(phn_mdl_files[i]))#, map_location= lambda stor, loc: stor))
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

result = train.ensemble_eval(phn_test_data, phn_mdls, phn_args, log_file_handle=log_file_handle)
print("Accuracy on Test: {1} for PHN".format(result))
print("Accuracy on Test: {1} for PHN".format(result), file=log_file_handle)
