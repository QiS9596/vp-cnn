"""
This is a script the run some bert embedding CNN test to make sure the module is runnable
"""
import vp_dataset_bert
import model_bert
import bert_train
import argparse
import numpy as np
import pandas as pd
import os
parser = argparse.ArgumentParser(description="Hyperparameter tunning for BERT-CNN(grid search)")
parser.add_argument('-class-num', type=int, default=361, help='number of different classes [default:361]')
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
parser.add_argument('-npy-post-fix', type=str, default='auto', help='npy file post fix for the script to look for target npy file [default:auto]')
parser.add_argument('-data-dir', type=str, default='data/bert_embeddings', help='path to dataset [default:data/bert_embeddings]')
parser.add_argument('-splitted', action='store_true', default=False, help='set if to load splitted dataset for cross-validation [default:False]')
parser.add_argument('-embed-dim', type=int, default=768, help='embedding dimensionality')
# TODO : integrate load splitted dataset in to script
# could be used to test cnn for bert embedding cnn, rcnn for dimensionality Reduced CNN, and aed for AutoEncoderDecoder
parser.add_argument('-model-mode', type=str, default='cnn', help='model to be tested, possible value[cnn; rcnn; aed], [default:cnn]')
# the pretrain aruguments are only used for rcnn and aed; for rcnn, the hyperparameter of it's sub aed model would be fixed at the minimum value
# if we are training autoencoder only, we will turn to grid
# parser.add_argument('-pretrain-lr', type=float, default=1e-2, help='pretrain learning rate for rcnn submodel [default:1e-2]')
parser.add_argument('-pretrain-lr-low', type=float, default=1e-3, help='grid search minimum learning rate for aed [defualt:1e-3]')
parser.add_argument('-pretrain-lr-high', type=float, default=2e-2, help='grid search maximum learning rate for aed [default:2e-2]')
parser.add_argument('-pretrain-lr-step', type=float, default=5e-4, help='grid search learning rate step size for aed [default:5e-4]')
parser.add_argument('-pretrain-batch-low', type=int, default=1, help='grid search min batch size for aed [default:1]')
parser.add_argument('-pretrain-batch-high', type=int, default=65, help='grid search max batch size for aed [defualt:65]')
parser.add_argument('-pretrain-batch-step', type=int, default=20, help='grid search batch size step [default:20]')
parser.add_argument('-pretrain-early-stop-loss-low', type=float, default=1e-2, help='min early stop loss for aed [default:1e-2]')
parser.add_argument('-pretrain-early-stop-loss-high', type=float, default=3e-1, help='max early stop loss for aed [default:3e-1]')
parser.add_argument('-pretrain-early-stop-loss-step', type=float, default=5e-3, help='early stop loss step for aed [default:5e-3]')
parser.add_argument('-pretrain-epochs-low', type=int, default=25, help='grid search min epochs for aed [default:25]')
parser.add_argument('-pretrain-epochs-high', type=int, default=1000, help='grid search max epochs for aed [default:1000]')
parser.add_argument('-pretrain-epochs-step', type=int, default=100, help='grid search step of epochs of aed [default:100]')
# parser.add_argument('-aed-layers', type=str, default='3072,1500,800,300', help='number of neurons in each layer in dense aed')

# possible embedding method: concat4, avg4, f1
parser.add_argument('-logdir', type=str, default='./data/result.csv', help='result dir for logging')
args = parser.parse_args()
# these are the things I don't want to add to the argument for now, but keep them here can making it easy to make it
# changable without changing the grid search loop
all_tsv_path = os.path.join(args.data_dir, 'all.tsv')
label_tsv_path = os.path.join(args.data_dir, 'labels.tsv')
if args.npy_post_fix == 'auto':
    npy_post_fix = args.embed_method
else:
    npy_post_fix = args.npy_post_fix
bert_data_npy = os.path.join(args.data_dir, 'all_'+npy_post_fix+'.npy')
bert_label_npy = os.path.join(args.data_dir, 'labels_'+npy_post_fix+'.npy')
possible_optimizers = ['adadelta']
validation_sum = 0.0
# we keep this possible combination of layers here, if we further want to search them it would be easy to refactor
dense_aed_layer_neurons_3072 = [
    [3072, 1500, 800, 300],
    [3072, 2000, 1000, 300],
    [3072, 2500, 2000, 1500, 1000, 300],
    [3072, 1200, 300]
]
dense_aed_layer_neurons_768 = [
    [768, 500, 300],
    [768, 300],
    [768, 600, 400, 300]
]
embed_dim_2_layer = {3072:dense_aed_layer_neurons_3072, 768:dense_aed_layer_neurons_768}
"""-------------previous script for try and error -----------------------------------------------------"""
# for i in range(10):
#     train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.splits(filename=all_tsv_path,
#                                                                    foldid=i,
#                                                                    label_filename=label_tsv_path,
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
def get_10fold_acc(n_kernels=500, lr=1e-3, epochs=1000, batch_size=50, optimizer='adadelta', embed_dim=768, class_num=361,
                   aed_layers=[768, 500, 300], pretrain_lr=1e-3, pretrain_batch_size=50, pretrain_early_stop_loss=1e-2,
                   pretrain_epochs=50):
    validation_sum = 0.0
    for i in range(10):
        if args.splitted:
            data_path = os.path.join(args.data_dir, str(i))
            train_tsv = os.path.join(data_path, 'train.tsv')
            train_npy = os.path.join(data_path, 'train.npy')
            eval_tsv = os.path.join(data_path, 'dev.tsv')
            eval_npy = os.path.join(data_path, 'dev.npy')
            train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.load_one_fold(train_tsv, train_npy, eval_tsv, eval_npy, class_num=class_num)
        else:
            train, dev, test = vp_dataset_bert.VPDataset_bert_embedding.splits(filename=all_tsv_path,
                                                                               foldid=i,
                                                                               label_filename=label_tsv_path,
                                                                               train_npy_name=bert_data_npy,
                                                                               label_npy_name=bert_label_npy,
                                                                               num_experts=0,
                                                                               embed_dim=args.embed_dim)
        if args.model_mode == 'cnn':
            mdl = model_bert.CNN_Embed(kernel_num=n_kernels, class_num = class_num, embed_dim=embed_dim)
            acc,model = bert_train.train_wraper(train_iter=train, dev_iter=dev, optimizer=optimizer, model=mdl, lr=lr, epochs=epochs, batch_size=batch_size)
            validation_acc = bert_train.eval_wraper(test, model, batch_size=50)
            validation_sum += validation_acc
        elif args.model_mode == 'rcnn':
            #TODO
            pass
        elif args.model_mode == 'aed':
            # train_ = vp_dataset_bert.AutoEncoderPretrainDataset.from_VPDataset_bert_embedding(train)
            mdl = model_bert.AutoEncoderDecoder(layers=aed_layers)
            loss, model = bert_train.train_wraper(train_iter=train, dev_iter=None, model=mdl,
                                                  pretrain_optimizer=optimizer, pretrain_lr=pretrain_lr,
                                                  pretrain_batch_size=pretrain_batch_size,
                                                  pretrain_early_stop_loss=pretrain_early_stop_loss,
                                                  pretrain_epochs=pretrain_epochs,
                                                  mode='auto_encoder_decoder')
            validation_loss = bert_train.eval_wraper(test, model, batch_size=50, mode='auto_encoder_decoder')
            validation_sum += validation_loss

    return validation_sum/10.0


if args.embed_method == 'concat4':
    embed_dim = 768*4
else:
    embed_dim = 768
embed_dim = args.embed_dim
result = []
if args.model_mode == 'cnn':
    for lr in np.arange(args.lr_low, args.lr_high+args.lr_step, args.lr_step):
        for epochs in range(args.epoch_low, args.epoch_high+1, args.epoch_step):
            for batch_size in range(args.batch_low, args.batch_high+1, args.batch_step):
                for n_kernels in range(args.nkernels_low, args.nkernels_high+1, args.nkernels_step):
                    for optimizer in possible_optimizers:
                        acc = get_10fold_acc(n_kernels=n_kernels,
                                             lr=lr,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             optimizer=optimizer,
                                             embed_dim=embed_dim,
                                             class_num=args.class_num
                                             )
                        result.append([n_kernels, lr, epochs, batch_size, optimizer, embed_dim, acc])
                        result_ = np.array(result)
                        df = pd.DataFrame(data=result_,
                                          columns=['n_kernels', 'lr', 'epochs', 'batch_size', 'optimizer', 'embed_dim', 'acc'])
                        df.to_csv(args.logdir)
if args.model_mode == 'aed':
    for lr in np.arange(args.pretrain_lr_low,
                        args.pretrain_lr_high+args.pretrain_lr_step,
                        args.pretrain_lr_step):
        for batch in range(args.pretrain_batch_low,
                           args.pretrain_batch_high + args.pretrain_batch_step,
                           args.pretrain_batch_step):
            for early_stop_loss in np.arange(args.pretrain_early_stop_loss_low,
                                             args.pretrain_early_stop_loss_high+args.pretrain_early_stop_loss_step,
                                             args.pretrain_early_stop_loss_step):
                for epoch in range(args.pretrain_epochs_low,
                                   args.pretrain_epochs_high+1,
                                   args.pretrain_epochs_step):
                    for optimizer in possible_optimizers:
                        for layer_option in embed_dim_2_layer[embed_dim]:
                            loss = get_10fold_acc(embed_dim=embed_dim,
                                                  aed_layers=layer_option,
                                                  pretrain_lr=lr,
                                                  pretrain_batch_size=batch,
                                                  pretrain_early_stop_loss=early_stop_loss,
                                                  pretrain_epochs=epoch,
                                                  optimizer=optimizer)
                            result.append([layer_option, lr, batch, early_stop_loss, epoch, optimizer,loss])
                            result_ = np.array(result)
                            df = pd.DataFrame(data=result,
                                              columns=['layer', 'lr', 'batch','early_stop', 'epoch','optimizer','loss'])
                            df.to_csv(args.logdir)
