import EmbeddingManager
import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('Get w2v embedding npy file')
parser.add_argument('-input-dir', type=str, default='./data/new', help='Directory for input file, looking for all.tsv and labels.tsv in it')
parser.add_argument('-output-dir', type=str, default='./data/bert_embeddings', help='Directory for output file, throw all_w2v.npy and labels_w2v.npy into it')
parser.add_argument('-w2v', type=str, default='../w2v/w2v.300d.txt', help='Path that lead to w2v.txt file')
args = parser.parse_args()

# initialize EmbeddingManager object
manager = EmbeddingManager.BERTEmbedManager()
manager.load_w2v_matrix(args.w2v)

# load, embed and save all data
all_tsv = os.path.join(args.input_dir, 'all.tsv')
all_tsv = pd.read_csv(all_tsv, sep='\t', header=None, names=['label', 'text'])

all_np = manager.w2v_embedding(all_tsv)
all_npy = os.path.join(args.output_dir, 'all_w2v.npy')
np.save(all_npy, all_np)
# load, embed and save labels data
labels_tsv = os.path.join(args.input_dir, 'labels.tsv')
labels_tsv = pd.read_csv(labels_tsv, sep='\t', header=None, names=['label','text'])

labels_np = manager.w2v_embedding(labels_tsv)
labels_npy = os.path.join(args.output_dir, 'lables_w2v.npy')
np.save(labels_npy, labels_np)