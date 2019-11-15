from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np


class VPDataset_bert_embedding(Dataset):
    """
    The object serves as dataset object to help load bert embedding model
    the function is achieved via wrapping a Pandas.DataFrame object
    """

    def __init__(self, df):
        """
        Initialization of dataset object
        :param df: Pandas.DataFrame object
        """
        self.df = df

    def __getitem__(self, index):
        return {'label':self.df.iloc[index]['labels'], 'embed':self.df.iloc[index]['embed']}

    def __len__(self):
        return len(list(self.df.index))

    @classmethod
    def splits(cls, num_fold=10, foldid=0, root='.', filename=None, label_filename=None, num_experts=5,
               dev_split=0.1):
        """
        This method splits the dataset into two parts: the training data and the testing data. We would not use
        development set to monitor the performance on unseen data during the training.
        If num_experts = 0 returns a tuple, the first element of the tuple would be the training dataset object,
        the second would be the development object, the third one would be the test object
        If num_experts > 0, returns a tuple; the first element of the tuple is a list object contains the dataset
        objects for training, the second element of the tuple is also a list object, contains the dataset objects for
        development, both of the list has the same length as num_experts; the third element is also a list object,
        containing the dataset object for test data obtained via cross validation
        To make each of the CNN classifier in the ensemble slightly different, we will assign slightly different training
        and development set to them.
        :param num_fold: int; number of fold
        :param foldid: int; id of current fold; the fold with foldid will be used as test set for cross validation,
        the other splits will be used for development and training
        :param root: str: root directory of the datafiles
        :param filename: str: file name of all data, should be tsv
        :param label_filename: str: file name of label embedding data, should be tsv
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
        df = pd.read_csv(data_path, sep='\t',header=None, names=['labels', 'embed'])
        df_label = pd.read_csv(label_path, sep='\t', header=None, names=['labels', 'embed'])
        # We split the data into k splits
        fold_dfs = np.array_split(ary=df,indices_or_sections=num_fold)
        # claim test fold
        test_ = cls(df=fold_dfs[foldid])
        # then concat the rest fold as train-dev data
        train_dev_folds = fold_dfs[:foldid] + fold_dfs[foldid + 1:]
        df_train_dev = pd.concat(train_dev_folds)
        dev_length = int(np.floor(dev_split * float(len(df_train_dev))))
        # if the num_experts == 0, then it indicates only one CNN model is trained
        if num_experts == 0:

            # we substract last several element correspond to the dev_split parameter and set them as development set

            dev_df = df_train_dev[-1*dev_length:]
            # the other training examples plus the label set are served as training set
            train_df = pd.concat([df_train_dev[:-1*dev_length], df_label])
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
                devs.append(cls(df=df_train_dev[dev_length*i:dev_length*(i+1)]))
                train_current = pd.concat([df_train_dev[:dev_length*i],
                                           df_train_dev[dev_length*(i+1):],
                                           df_label])
                trains.append(cls(df=train_current))
            return (trains, devs, test_)

