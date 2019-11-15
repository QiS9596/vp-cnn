"""
This is a script the run some bert embedding CNN test to make sure the module is runnable
"""
import vp_dataset_bert
import model_bert
import bert_train
bert_embedding_path = 'data/bert_embeddings/all.tsv'
bert_label_embedding_path = 'data/bert_embeddings/labels.tsv'
train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.split(filename=bert_embedding_path,
                                         label_filename=bert_label_embedding_path,
                                         num_expert=0)
model_cnn = model_bert.CNN_Embed()
bert_train.train(train=train, dev=None,model=model_cnn)