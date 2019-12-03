"""
This is a script the run some bert embedding CNN test to make sure the module is runnable
"""
import vp_dataset_bert
import model_bert
import bert_train
import argparse
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser(description="Hyperparameter tunning for BERT-CNN(grid search)")
parser.add_argument('-lr-low', type=float, default=1e-3, help='minimum value of learning rate [default:1e-3]')
parser.add_argument('-lr-high', type=float, default=1e-2, help='maximum value of learning rate [default:1e-2]')
parser.add_argument('-lr-step', type=float, default=5e-4, help='step size for learning rate search [default:5e-4]')
parser.add_argument('-epoch-low', type=int, default=500, help='minimum number of epochs [default:500]')
parser.add_argument('-epoch-high', type=int, default=3000, help='maximum number of epochs [default:3000]')
parser.add_argument('-epoch-step', type=int, default=250, help='step size of epoch search [default:250]')
parser.add_argument('-batch-low', type=int, default=1, help='minimum number of batch size [default:1]')
parser.add_argument('-batch-high', type=int, default=300, help='maximum number of batch size [default:1]')
parser.add_argument('-batch-step', type=int, default=25, help='step size for batch size search [default:25]')
parser.add_argument('-nkernels-low', type=int, default=50, help='minimum number of each type of filters [default:50]')
parser.add_argument('-nkernels-high', type=int, default=800, help='maximum number of each type of filters [default:800]')
parser.add_argument('-nkernels-step', type=int, default=50, help='step size of number of each type of filters [default:50]')
parser.add_argument('-embed-method', type=str, default='avg4', help='method of extracting embedding from BERT [default:avg4]')
parser.add_argument('-logdir', type=str, default='./data/result.csv', help='result dir for logging')
args = parser.parse_args()
# these are the things I don't want to add to the argument for now, but keep them here can making it easy to make it
# changable without changing the grid search loop
bert_embedding_path = 'data/bert_embeddings/all.tsv'
bert_label_embedding_path = 'data/bert_embeddings/labels.tsv'
bert_data_npy = 'data/bert_embeddings/all_avg4.npy'
bert_label_npy = 'data/bert_embeddings/labels_avg4.npy'
possible_optimizers = ['adadelta']
validation_sum = 0.0
"""-------------previous script for try and error -----------------------------------------------------"""
# for i in range(10):
#     train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.splits(filename=bert_embedding_path,
#                                                                    foldid=i,
#                                                                    label_filename=bert_label_embedding_path,
#                                                                    train_npy_name=bert_data_npy,
#                                                                    label_npy_name=bert_label_npy,
#                                                                    num_experts=0)
#     model_cnn = model_bert.CNN_Embed(kernel_num=500, embed_dim=768)
#     acc, model = bert_train.train(train=train, dev=dev,optimizer='adadelta',model=model_cnn,lr=5e-3, epochs=1000, batch_size=50)
#     validation_acc = bert_train.eval(test, model, batch_size=50)
#     validation_sum += validation_acc
# print("train seems complete")
# print(validation_sum/10.0)
"""--------------previous script ended-----------------------------------------------------------------"""
# function of ten fold for a single set of hyper-parameters
def get_10fold_acc(n_kernels=500, lr=1e-3, epochs=1000, batch_size=50, optimizer='adadelta', embed_dim=768):
    validation_sum = 0.0
    for i in range(10):
        train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.splits(filename=bert_embedding_path,
                                                                           foldid=i,
                                                                           label_filename=bert_label_embedding_path,
                                                                           train_npy_name=bert_data_npy,
                                                                           label_npy_name=bert_label_npy,
                                                                           num_experts=0)
        mdl = model_bert.CNN_Embed(kernel_num=n_kernels,embed_dim=embed_dim)
        acc,model = bert_train.train(train=train, dev=dev, optimizer=optimizer, model=mdl, lr=lr, epochs=epochs, batch_size=batch_size)
        validation_acc = bert_train.eval(test, model, batch_size=50)
        validation_sum += validation_acc
    return validation_sum/10.0
if args.embed_method == 'concat4':
    embed_dim = 768*4
else:
    embed_dim = 768
result = []
for lr in np.array_split(args.lr_low, args.lr_high+args.lr_step, args.lr_step):
    for epochs in range(args.epoch_low, args.epoch_high+1, args.epoch_step):
        for batch_size in range(args.batch_low, args.batch_high+1, args.batch_step):
            for n_kernels in range(args.nkernels_low, args.nkernels_high+1, args.nkernels_step):
                for optimizer in possible_optimizers:
                    acc = get_10fold_acc(n_kernels=n_kernels,
                                         lr=lr,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         optimizer=optimizer,
                                         embed_dim=embed_dim
                                         )
                    result.append([n_kernels, lr, epochs, batch_size, optimizer, embed_dim, acc])
                    result_ = np.array(result)
                    df = pd.DataFrame(data=result_,
                                      columns=['n_kernels', 'lr', 'epochs', 'batch_size', 'optimizer', 'embed_dim', 'acc'])
                    df.to_csv(args.logdir)
