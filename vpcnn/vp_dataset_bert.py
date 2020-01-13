import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VPDataset_bert_embedding(Dataset):
    """
    The object serves as dataset object to help load bert embedding model
    the function is achieved via wrapping a Pandas.DataFrame object
    """

    def __init__(self, df, use_dummies=False):
        """
        Initialization of dataset object
        :param df: Pandas.DataFrame object
        """
        self.df = df
        if use_dummies:
            label_dum = pd.get_dummies(self.df['labels']).values
            dum_in_list = []
            for i in range(label_dum.shape[0]):
                dum_in_list.append(label_dum[i, :])
            self.df['labels'] = dum_in_list

    def __getitem__(self, index):
        return {'label': self.df.iloc[index]['labels'], 'embed': self.df.iloc[index]['embed']}

    def __len__(self):
        return len(list(self.df.index))

    @staticmethod
    def load_npy_and_tsv(tsv_path='./data/all.tsv', npy_path='./data/bert_embeddings/all.npy'):
        """
        Load the training labels from tsv files, load the embeddings from npy files then merge them together
        The returned Pandas.DataFrame object has two column, the first one is 'label' for the training labels,
        the second one is 'embed' for the word embeddings
        :param tsv_path: str: path to the label/text dataset
        :param npy_path: str: path to the wordembedding np.array
        :return: Pandas.DataFrame object
        """
        df_ = pd.read_csv(tsv_path, sep='\t', header=None, names=['label', 'text'])
        matrix = np.load(npy_path, allow_pickle=True)
        assert len(list(df_.index)) == matrix.shape[0]
        df_.drop(['text'], axis=1)
        df_['embed'] = matrix
        return df_

    @staticmethod
    def sequence_padding(df, max_seq_len=32):
        """
        pad the sequence of the 'embed' field for the target dataframe object.
        each element of embed field should be a np.array object, which has two dimensionalities N*M; representing N
        The method will cut the embeddings if the sentence has more than max_seq_len number of words; it will also
        add array matrix filled with 0 at the end if the length of sentence is smaller than max_seq_len
        words in a sentence and each word has bert embedding at M dimension
        :param df: Pandas.DataFrame object, should contain a column called 'embed'
        :param max_seq_len: int; maximum sequence length
        :return: Pandas.DataFrame, the dataframe object after processing
        """
        num_example = len(df.index)
        # handle each row
        for row_idx in range(num_example):
            current_row_len = len(df['embed'][row_idx])
            # if the current row has the same length as the max sequence length, do nothing
            if current_row_len == max_seq_len:
                continue
            # if the current row has longer sequence, then cut
            elif current_row_len > max_seq_len:
                df['embed'][row_idx] = df['embed'][row_idx][:max_seq_len, :]
            # if the current row has shorter sequence, then pad
            else:
                df['embed'][row_idx] = np.pad(df['embed'][row_idx],
                                              ((0, max_seq_len - current_row_len), (0, 0)),
                                              'constant',
                                              constant_values=[0])
        return df

    @classmethod
    def splits(cls, num_fold=10, foldid=0, root='.', filename=None, label_filename=None, train_npy_name=None,
               label_npy_name=None, num_experts=5, dev_split=0.1, max_seq_len=32):
        """
        This method splits the dataset into two parts: the training data and the testing data. We would not use
        development set to monitor the performance on unseen data during the training.
        The data we use come from two parts: the collected data and the label data. The label data is one example for
        each of the class, adding it to each fold to make sure each of these folds has at least one example for each
        class.
        If num_experts = 0 returns a tuple, the first element of the tuple would be the training dataset object,
        the second would be the development object, the third one would be the test object
        If num_experts > 0, returns a tuple; the first element of the tuple is a list object contains the dataset
        objects for training, the second element of the tuple is also a list object, contains the dataset objects for
        development, both of the list has the same length as num_experts; the third element is also a list object,
        containing the dataset object for test data obtained via cross validation
        To make each of the CNN classifier in the ensemble slightly different, we will assign slightly different training
        and development set to them.
        Since saving pandas.DataFrame will corrupt the high dimensional np.array, the word embeddings are saved in the
        corresponding npy file.
        :param num_fold: int; number of fold
        :param foldid: int; id of current fold; the fold with foldid will be used as test set for cross validation,
        the other splits will be used for development and training
        :param root: str: root directory of the datafiles
        :param filename: str: file name of all data, should be tsv
        :param label_filename: str: file name of label embedding data, should be tsv
        :param train_npy_name: file name of npy file for all/training data
        :param label_npy_name: file name of npy file for label/supporting data
        :param num_experts: int; number of expert in ensemble
        :param dev_split: float; the relative size of development set compared to
        :return: a tuple of three elements
        """
        # obtaining the entire list is currently not supported
        if foldid is None:
            raise NotImplementedError
        # load the data
        data_path = os.path.join(root, filename)
        label_path = os.path.join(root, label_filename)
        data_npy_path = os.path.join(root, train_npy_name)
        label_npy_path = os.path.join(root, label_npy_name)
        df = pd.read_csv(data_path, sep='\t', header=None, names=['labels', 'embed'])
        df_label = pd.read_csv(label_path, sep='\t', header=None, names=['labels', 'embed'])

        npy_data = np.load(data_npy_path, allow_pickle=True)
        npy_label = np.load(label_npy_path, allow_pickle=True)
        df = df['labels'].to_frame()
        df['embed'] = npy_data
        df_label = df_label['labels'].to_frame()
        df_label['embed'] = npy_label
        df = VPDataset_bert_embedding.sequence_padding(df, max_seq_len=max_seq_len)
        df_label = VPDataset_bert_embedding.sequence_padding(df_label, max_seq_len=max_seq_len)
        # We split the data into k splits
        fold_dfs = np.array_split(ary=df, indices_or_sections=num_fold)
        # claim test fold
        test_ = cls(df=fold_dfs[foldid])
        # then concat the rest fold as train-dev data
        train_dev_folds = fold_dfs[:foldid] + fold_dfs[foldid + 1:]
        df_train_dev = pd.concat(train_dev_folds)
        dev_length = int(np.floor(dev_split * float(len(df_train_dev))))
        # if the num_experts == 0, then it indicates only one CNN model is trained
        if num_experts == 0:

            # we substract last several element correspond to the dev_split parameter and set them as development set

            dev_df = df_train_dev[int(-1 * dev_length):]
            # the other training examples plus the label set are served as training set
            train_df = pd.concat([df_train_dev[:int(-1 * dev_length)], df_label])
            return (
                cls(df=train_df),
                cls(df=dev_df),
                test_)
        else:
            # assert that each stack of CNN contains at most 5 agents
            assert num_experts <= 5
            devs = []
            trains = []

            # then for each expert, it will gets a dev set and a training set, the length of dev sets are all dev_length
            # but the dev set and training set for each expert are different to make sure they are divergent
            for i in range(num_experts):
                devs.append(cls(df=df_train_dev[int(dev_length * i):int(dev_length * (i + 1))]))
                train_current = pd.concat([df_train_dev[:int(dev_length * i)],
                                           df_train_dev[int(dev_length * (i + 1)):],
                                           df_label])
                trains.append(cls(df=train_current))
            return (trains, devs, test_)


class AutoEncoderPretrainDataset(Dataset):
    """
    This object works as a Dataset object for pretraining an auto-encoder-decoder mentioned in model_bert.py:
    BaseAutoEncoderDecoder
    """

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        return {'embed': self.df.iloc[index]['embed']}

    def __len__(self):
        return len(list(self.df.index))

    @classmethod
    def from_VPDataset_bert_embedding(cls, vpdataset):
        """
        Takes a VPDataset_bert_embedding, and collapse the sequences of word embeddings into a collection of word
        embeddings, then emcapsulate with AutoEncoderPretrainDataset
        :param vpdataset: an object of VPDataset_bert_embedding
        :return: an object of AutoEncoderPretrainDataset
        """
        sequence_level_df = vpdataset.df
        embeddings = sequence_level_df['embed']  # Pandas.Series that contains the embeddings
        # we'll use a loop to traverse the dataframe to collapse the sentence structure
        # which might not be a best implementation
        word_embeddings = []
        for index in list(embeddings.index):
            # for each sentence we destroy the sentence structure and extends to the sequence of just embedding

            word_embeddings += embeddings.get(index).tolist()
        word_embeddings_df = pd.DataFrame()
        word_embeddings_df['embed'] = word_embeddings
        return cls(word_embeddings_df)
