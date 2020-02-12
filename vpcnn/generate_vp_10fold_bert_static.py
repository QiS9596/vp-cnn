import numpy as np
import EmbeddingManager
import pandas as pd
import argparse
import os
import time

parser = argparse.ArgumentParser("generate virtual patient average bert embedding training folds")
parser.add_argument('-tar-dir', type=str, default='./data/avgbertembed', help='dir for storing generated files')
parser.add_argument('-db', type=str, default='./data/bert_embed.db', help='database file')
parser.add_argument('-data-tsv', type=str, default='./data/new/new_all_nl.tsv',
                    help='data tsv file, should contain label and natural language sentence')
parser.add_argument('-label-tsv', type=str, default='./data/new/new_labels_nl.tsv',
                    help='label context for each class that support training process')
parser.add_argument('-layer', type=str, default='-12', help='bert layer for embedding extraction')
parser.add_argument('-n-folds', type=int, default=10, help='number of folds of the dataset')

args = parser.parse_args()
layer = args.layer
df_all = pd.read_csv(args.data_tsv, sep='\t', header=None, names=['label', 'text'])
df_label = pd.read_csv(args.label_tsv, sep='\t', header=None, names=['label', 'text'])

embedding_manager = EmbeddingManager.BERTEmbedManager(db_file=args.db)
# create n folds index
n_split_index = np.array_split(ary=list(df_all.index), indices_or_sections=args.n_folds)
start = time.time()
for i in range(args.n_folds):
    val_index = n_split_index[i]
    # select i-th split as validation set
    train_index = []
    for ii in range(args.n_folds):
        if not ii == i:
            train_index += n_split_index[ii].tolist()
    embed_dict, train_array, dev_array = embedding_manager.embed_average_bert(tsv_file=df_all,
                                                                              train_line_ids=train_index,
                                                                              layer=layer,
                                                                              dataset_name='VP1617STRICT',
                                                                              supplimentary_tarining_set='VP1617STRICTLABEL',
                                                                              sup_df=df_label)
    fold_id_path = os.path.join(args.tar_dir, str(i))
    os.makedirs(fold_id_path)
    train_df = df_all.ix[train_index]
    train_df = pd.concat([train_df, df_label])
    train_df.to_csv(os.path.join(fold_id_path, 'train.tsv'), sep='\t', header=False, index=False)
    np.save(os.path.join(fold_id_path, 'train.npy'), train_array)

    val_df = df_all.ix[val_index]
    val_df.to_csv(os.path.join(fold_id_path, 'dev.tsv'), sep='\t', header=False, index=False)
    np.save(os.path.join(fold_id_path, 'dev.npy'), dev_array)
    used = time.time()-start
    estimated = used/(i+1)*args.n_folds
    print("BERT embedding generating script: {} th fold completed, used {} sec, estimated time to be spent is {}".format(i, used, estimated))

