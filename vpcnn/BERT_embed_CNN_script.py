"""
This is a script the run some bert embedding CNN test to make sure the module is runnable
"""
import vp_dataset_bert
import model_bert
import bert_train
bert_embedding_path = 'data/bert_embeddings/all.tsv'
bert_label_embedding_path = 'data/bert_embeddings/labels.tsv'
bert_data_npy = 'data/bert_embeddings/all.npy'
bert_label_npy = 'data/bert_embeddings/labels.npy'
validation_sum = 0.0
for i in range(10):
    train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.splits(filename=bert_embedding_path,
                                                                   foldid=i,
                                                                   label_filename=bert_label_embedding_path,
                                                                   train_npy_name=bert_data_npy,
                                                                   label_npy_name=bert_label_npy,
                                                                   num_experts=0)
    model_cnn = model_bert.CNN_Embed(kernel_num=500)
    acc, model = bert_train.train(train=train, dev=dev,optimizer='adadelta',model=model_cnn,lr=1e-2, epochs=500, batch_size=50)
    validation_acc = bert_train.eval(test, model, batch_size=50)
    validation_sum += validation_acc
print("train seems complete")
print(validation_sum/10.0)
