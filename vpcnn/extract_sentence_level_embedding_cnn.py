"""
This is a script that train a CNN model using CNN model (which directly takes word embeddings) on the entire dataset.
Then we extract the sentence level embedding of the training set learnt by such CNN model for further analysis
Here, the sentence level representation of CNN model refers to the hidden state that is feed into last fully connected
layer.
"""

import vp_dataset_bert
import bert_train
import model_bert
import argparse
import numpy as np
import pandas as pd
import os
import torch

parser = argparse.ArgumentParser(description="Train and extracting sentence level representation of CNN model")
parser.add_argument('-class-num', type=int, default=334, help='number of different classes [default:334]')
# the learning rate hyper-parameter is not used since we are going to use adadelta with fixed initial learning rate and
# fixed rho
parser.add_argument('-epoch', type=int, default=50,
                    help='number of epochs for training the CNN classifier [default:50]')
parser.add_argument('-batch', type=int, default=100, help='size of batch for training the CNN classifier [default:100]')
parser.add_argument('-nkernels', type=int, default=500, help='number of kernels for each size [defualt:500]')
parser.add_argument('-embed-method', type=str, default='f1',
                    help='method of extracting embedding from BERT [default:f1]')
parser.add_argument('-npy-post-fix', type=str, default='auto',
                    help='post fix to look for npy file for word embedding [defualt: auto] looking for the one with same' \
                         'name as embed method')
parser.add_argument('-data-dir', type=str, default='./data/bert_embeddings', help='path to find input data')
parser.add_argument('-embed-dim', type=int, default=768, help='embedding dimensionality for each word [default:768]')
parser.add_argument('-max-seq', type=int, default=32, help='max sequence length for sequence padding')

parser.add_argument('-log-file', type=str, default='./data/sentence_embedding',
                    help='result file name prefix [default:./data/sentence_embedding ' \
                         'will produce two file, $log_file$.csv and $log_file$.npy ' \
                         'the csv file will contain the original example, the expected output and and actual output' \
                         'the npy file should contain the sentence level embedding'
                    )

args = parser.parse_args()

# load dataset
# the data_dir should contain four files, including all.tsv, all_$npy_post_fix$.npy, labels.tsv,
# and labels_$npy_post_fix$.npy
all_tsv_path = os.path.join(args.data_dir, 'all.tsv')
label_tsv_path = os.path.join(args.data_dir, 'labels.tsv')
if args.npy_post_fix == 'auto':
    npy_post_fix = args.embed_method
else:
    npy_post_fix = args.npy_post_fix
bert_data_npy = os.path.join(args.data_dir, 'all_' + npy_post_fix + '.npy')
bert_label_npy = os.path.join(args.data_dir, 'labels_' + npy_post_fix + '.npy')

df_all = pd.read_csv(all_tsv_path, sep='\t', header=None, names=['labels', 'text'])
df_labels = pd.read_csv(label_tsv_path, sep='\t', header=None, names=['labels', 'text'])
npy_all = np.load(bert_data_npy, allow_pickle=True)
npy_label = np.load(bert_label_npy, allow_pickle=True)

df_all['embed'] = npy_all
df_labels['embed'] = npy_label

df_all = vp_dataset_bert.VPDataset_bert_embedding.sequence_padding(df_all, max_seq_len=args.max_seq, embed_dim=args.embed_dim)
df_labels = vp_dataset_bert.VPDataset_bert_embedding.sequence_padding(df_labels, max_seq_len=args.max_seq, embed_dim=args.embed_dim)

df_train = pd.concat([df_all, df_labels])
dataset_train = vp_dataset_bert.VPDataset_bert_embedding(df=df_train)
# claim and train model
mdl = model_bert.CNN_Embed(kernel_num=args.nkernels, class_num=args.class_num, embed_dim=args.embed_dim)
acc, model = bert_train.train_wraper(train_iter=dataset_train, dev_iter=dataset_train, optimizer='adadelta', model=mdl,
                                     epochs=args.epoch, batch_size=args.batch)
result = []
# predict and extract representation
for index in list(df_all.index):
    x = df_all.ix[index]['embed']
    sentence_level_representation = mdl.sentence_level_representation(x)
    logit = mdl(x)
    probs, predicted = torch.max(torch.exp(logit), 1)
    predicted = predicted.cpu().data.numpy()
    result.append([df_all.ix[index]['text'], df_all.ix[index]['labels'], predicted, sentence_level_representation])
# formatting and output
result_df = pd.DataFrame(data=result,
                  columns=['text', 'labels', 'predicted', 'sentence_level_embed'])
sle = result_df['sentence_level_embed'].to_frame().to_numpy()
result_df = result_df.drop(['sentence_level_embed'], axis=1)
sle_npy = args.log_file + '.npy'
result_csv = args.log_file + '.csv'
np.save(sle_npy, sle)
result_df.to_csv(result_csv)
