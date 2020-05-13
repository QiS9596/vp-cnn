"""
This script runs CNN classifier on weighted sum of BERT embeddings across representations extracted from multiple layer
of BERT.
"""

# we isolated the following varible, so that in the future it would be easier to augment this script
import vp_dataset_bert
import os
import model_bert

# path and name to related data file
# the label and original question is stored in the tsv file
# however the embedding is stored in the npy file
# we have the all file that contains the sentences collected from medical school student interacting with ChatScript
# agent, which will devided into folds
# then we have the label file in which there is the context for each label, this portion will be added to the training
# set in each fold
data_dir = './data/new_data'
all_tsv_name = 'new_all_nl.tsv'
label_tsv_name = 'new_labels_nl.tsv'
all_npy_name = 'all_12_reshaped.npy'
label_npy_name = 'labels_12_reshaped.npy'

xfolds = 10

def one_fold(fold_id=0):
    train, dev, test = vp_dataset_bert.WeightedSumBERTCNNDataset.splits(num_fold=xfolds,
                                                                        foldid=fold_id,
                                                                        all_df_name=os.path.join(data_dir,all_tsv_name),
                                                                        all_npy_name=os.path.join(data_dir, all_npy_name),
                                                                        label_df_name=os.path.join(data_dir, label_tsv_name),
                                                                        label_npy_name=os.path.join(data_dir, label_npy_name))
    mdl = model_bert.WeightedSumEmbedding()
    mdl.train_mdl(train, dev, mdl)
    mdl.eval_mdl(test, mdl, batch_size=40)

one_fold()
