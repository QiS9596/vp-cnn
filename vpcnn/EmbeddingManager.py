import copy
import os
import random
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition as decomp
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import seaborn as sns

Axes3D
import token_db
import tokenization
from string import punctuation

class BERTEmbedManager:
    """
    This class provide functionality to do any fancy (or not fancy) stuff to generate or process BERT embeddings for
    given input

    Some of the primary solution we wish to provide via this class
    Clustering with measurement on BERT embeddings
    Average of BERt embeddings
    generate correspond NPY file for training
    """

    def __init__(self, db_file='./data/bert_embed.db', bert_vocab='./data/bert_embeddings/vocab.txt'):
        self.db_manager = token_db.BERTDBProcessor(db_file)
        self.tokenizer = tokenization.FullTokenizer(bert_vocab)
        self.dfs = {}
        self.w2v_matrix = None

    def load_nl_df(self, path, dataset_name, sep='\t', header=None, names=['label', 'text']):
        """
        load target dataframe into memory, store it into self.dfs
        :param path:
        :param dataset_name:
        :param sep:
        :param header:
        :param names:
        :return:
        """
        df = pd.read_csv(path, sep=sep, header=header, names=names)
        self.dfs[dataset_name] = df

    def clean_df(self, dataset_name):
        """
        Delete target df in self.dfs to free up memory
        :param dataset_name: name of the dataset
        :return:
        """
        try:
            self.dfs.pop(dataset_name)
        except KeyError:
            return

    def load_w2v_matrix(self, path):
        """
        Load target w2v matrix
        :param path: path to the w2v.txt file
        :return:
        """
        if self.w2v_matrix is None:
            self.w2v_matrix = {}
        with open(path) as file:
            for line in file.readlines():
                splitted = line.split(' ')
                token = splitted[0]
                embed = [float(i) for i in splitted[1:]]
                self.w2v_matrix[token] = np.array(embed)

    def w2v_embedding(self, tsv_file):
        """
        use w2v embedding to embed the data.
        the w2v embedding matrix should be loaded into this embedding manager object via load_w2v_matrix
        the tokenization pipeline for a sentence is merely remove the punctuations, shift to lower case and split on
        spaces
        if OOV is found, use a random word to replace

        :param tsv_file: the tsv file of the texts to be transformed
        :return: a numpy.ndarray or None
        """
        if self.w2v_matrix is None:
            return None
        if isinstance(tsv_file, str):
            df = pd.read_csv(tsv_file, sep='\t', header=None, names=['label', 'text'])
        else:
            df = tsv_file
        result = []
        for sentence_idx in range(len(list(df.index))):
            sentence = df.iloc[sentence_idx]['text']
            tokens = self.simple_tokenize(sentence)
            sentence_embed = []
            for token in tokens:
                try:
                    embed = self.w2v_matrix[token]
                except KeyError:
                    random_token = random.choice(list(self.w2v_matrix.keys()))
                    embed = self.w2v_matrix[random_token]
                sentence_embed.append(np.array(embed))
            result.append(np.array(sentence_embed))
        return np.array(result)




    def simple_tokenize(self, sentence):
        """
        The simplest version of tokenization, used in getting the token embedding with w2v static embeddings
        :param sentence: string ,the input sentence
        :return: list of strings, each string is a token
        """
        sentence = sentence.lower()
        sentence = self.rmv_punc(sentence)
        return sentence.split(' ')

    def embed_average_bert(self, tsv_file, train_line_ids, layer, dataset_name, supplimentary_tarining_set=None,
                           sup_df=None):
        start = time.time()
        if isinstance(tsv_file, str):
            df = pd.read_csv(tsv_file, sep='\t', header=None, names=['label', 'text'])
        else:
            df = tsv_file
        if supplimentary_tarining_set:
            datasets = [dataset_name, supplimentary_tarining_set]
            line_selection = [train_line_ids, None]
            if isinstance(sup_df, str):
                sup = pd.read_csv(sup_df, sep='\t', header=None, names=['label', 'text'])
            else:
                sup = sup_df
        else:
            datasets = [dataset_name]
            line_selection = [train_line_ids]
        train_result = []
        embed_dict = {}
        # in the training corpus we build a quick check dictionary of average bert embedding
        # when encountering a new word, we first check it out in the dictionary for quick check
        # if it's not there then check the database. In theory we will not have OOV
        line_count = 0
        for i in train_line_ids:
            sentence = self.tokenizer.tokenize(df.iloc[i]['text'])
            sentence_embed = []
            for word in sentence:
                word = self.db_manager.rpl_punc(word)
                try:
                    sentence_embed.append(embed_dict[word])
                except KeyError:
                    avg_embed = self.db_manager.get_avg_bert_embed(datasets_name=datasets,
                                                                   token_name=word,
                                                                   line_selection=line_selection,
                                                                   layer=layer)
                    sentence_embed.append(avg_embed)
                    embed_dict[word] = avg_embed
            train_result.append(sentence_embed)
            if line_count % 50 == 0:
                used = time.time() - start
                estimated = used / (line_count + 1) * len(train_line_ids)
                print("Embedding Manager: {}/{} for current batch, used {} sec, estimated time to be spent is {}"
                      .format(line_count, len(train_line_ids), used, estimated))
            line_count += 1
        # then load the supplimentary training set and add it to the tail of the training embedding list
        if supplimentary_tarining_set:
            for sentence in sup['text']:
                sentence_embed = []
                for word in self.tokenizer.tokenize(sentence):
                    word = self.db_manager.rpl_punc(word)
                    try:
                        sentence_embed.append(embed_dict[word])
                    except KeyError:
                        avg_embed = self.db_manager.get_avg_bert_embed(datasets_name=datasets,
                                                                       token_name=word,
                                                                       line_selection=line_selection,
                                                                       layer=layer)
                        sentence_embed.append(avg_embed)
                        embed_dict[word] = avg_embed
                train_result.append(sentence_embed)
        # in the validation set, we build the embedding based on the dictionary we generated in the previous step
        # if OOV is found, simply randomly choose any one of the word in the dictionary as a replacement
        dev_result = []
        # get a list copy of collected tokens for randomly replacement of OOV
        collected_tokens = list(embed_dict.keys())
        for i in list(df.index):
            if i not in train_line_ids:
                sentence = self.tokenizer.tokenize(df.iloc[i]['text'])
                sentence_embed = []
                for word in sentence:
                    word = self.db_manager.rpl_punc(word)
                    try:
                        sentence_embed.append(embed_dict[word])
                    except KeyError:
                        # handle OOV
                        random_word = random.choice(collected_tokens)
                        sentence_embed.append(embed_dict[random_word])
                dev_result.append(sentence_embed)
        return (embed_dict, train_result, dev_result)

    def silhouetteplot(self, data, cluster_label, n_clusters, scatter_loc, title='', show=True, output=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax_ = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_xlim([-1.0, 1.0])
        vertical_spacing = (n_clusters + 1) * 50
        ax.set_ylim([-20, len(data) + vertical_spacing])
        average = silhouette_score(data, cluster_label)
        sample_silhouette_values = silhouette_samples(data, cluster_label)

        y_lower = 10

        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_label == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color_map = cm.get_cmap('nipy_spectral')
            color = color_map(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0,
                             ith_cluster_silhouette_values,
                             facecolor=color,
                             edgecolor=color,
                             alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 35
        ax.set_title('Silhouette Plot')
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Cluster Label')
        ax.axvline(x=average, color='red', linestyle='--')
        ax.set_yticks([])
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        color_map = cm.get_cmap('nipy_spectral')
        colors = color_map(cluster_label.astype(float) / n_clusters)
        ax_.scatter(scatter_loc[:, 0], scatter_loc[:, 1], scatter_loc[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        ax_.set_title('Reduced dimension data')
        plt.suptitle(title)
        if output:
            plt.savefig(output)
        if show:
            plt.show()
        fig.clear()
        plt.close(fig)

    def line_plot_silhouette(self, n_cluster_eval, title='', show=True, output=None):
        """"""
        x = [n[0] for n in n_cluster_eval]
        y = [n[1] for n in n_cluster_eval]
        plt.plot(x, y)
        plt.title(title)
        if output:
            plt.savefig(output)
        if show:
            plt.show()
        plt.close()


    def clustering_analysis_BERT_token_kmeans(self, dataset_name, token_name, layer, n_clusters_range=[2, 15, 2],
                                              trials=30, plot_fig=False, dimen_reduction_method=None,
                                              visualize_reduction_method='auto', output=None, possible_clusters='auto'):
        """
        This method apply k-mean cluster algorithm to a certain collection of bert embedding of target token to study
        it's behavior.
        If the dimen_rduction_method is None, then the clustering algorithm will not be applied before the clustering.
        The visualization_reduction_method is the dimensionality reduction method that is applied for better
        visualization the dataset. if 'auto' is selected, we will use pca when dien_reduction_method is None, or used
        the same method for dimen_reduction_method.
        The method will return necessary information, including the best number of clusters, the best clustering object,
        the best silhouette score, and the silhouette for best trials of each n-cluster clusterings.
        The method can also be used to plot the silhouette plot for the cluster combined with the scatter of the
        embeddings after dimensional reduction. Plus, to study the posibility of different clustering of the embeddings,
        a line plot of silhouette score of different n-clusers.

        :param dataset_name: str; name of the dataset
        :param token_name: str; token
        :param layer: str or int; refers to the target BERT layer for embedding extraction
        :param n_clusters_range: list of integers, used for search the best cluster number of k-means cluster, follow
        the pattern of [start, end, step] as the range function of python
        :param plot_fig: bool, if to use silhouette plot to plot figure
        :param dimen_reduction_method: method for dimensionality reduction
        :param visualize_reduction_method: dimensionality reduction for visualization purpose
        :param output: str; path for output; if it's None, do not output; otherwise output the figure into target path
        :param possible_clusters: str; if to plot the silhouette score for each possible cluster trial
        :return:
        """
        # collect the embeddings
        token_records = self.db_manager.get_embeddings_onLayer(dataset_name, token_name, layer)
        token_embeddings = []
        for token_record in token_records:
            token_embeddings.append(np.loads(token_record[0]))
        print(np.array(token_embeddings).shape)
        best_silhouette_score = -1.1
        best_clustering = None
        best_n_clusters = None
        n_clusters_eval = []
        # try different value of n_clusters, for each n cluster perform kmeans clustering
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1], n_clusters_range[2]):
            clustering = self.KMeans_clustering_bert_token(token_embeddings, n_clusters, trials=trials)
            # calculate the average silhouette score for the current clustering
            # record the silhouette measurement with the correspond n-cluster
            average_silhouette_score = silhouette_score(token_embeddings, clustering.labels_)
            n_clusters_eval.append((n_clusters, average_silhouette_score))
            # record the cluster with the highest silhouette score
            if average_silhouette_score > best_silhouette_score:
                best_silhouette_score = average_silhouette_score
                best_clustering = copy.deepcopy(clustering)
                best_n_clusters = n_clusters
        if plot_fig or output:
            title = 'Kmeans clustering on ' + dataset_name + ' ' + token_name
            if visualize_reduction_method == 'auto' and dimen_reduction_method is None:
                reduced = decomp.PCA(n_components=3).fit_transform(token_embeddings)
            if output == None:
                output = None
                if dimen_reduction_method is not None:
                    title += ' with dimensional reduction method ' + dimen_reduction_method
                if possible_clusters == 'auto':
                    title_n_clusters = title + ' n-clusters silhouette'
                    output_n_clusters = None
            else:
                name = dataset_name + '_' + token_name + '_' + str(layer) + '_kmeans_n-clusters' + str(best_n_clusters)
                if dimen_reduction_method is not None:
                    name += '_dimen-reduction' + dimen_reduction_method
                    title += ' with dimensional reduction method ' + dimen_reduction_method
                if possible_clusters == 'auto':
                    title_n_clusters = title + ' n-clusters silhouette'
                    name_n_clusters = name + '_n_clusters_silhouette.png'
                if visualize_reduction_method == 'auto' and dimen_reduction_method is None:
                    name += '_visual-reductionPCA'
                name += '.png'
                output_n_clusters = os.path.join(output, name_n_clusters)
                output = os.path.join(output, name)
            self.silhouetteplot(token_embeddings, best_clustering.labels_, best_n_clusters, reduced, title=title,
                                show=plot_fig, output=output)
            if possible_clusters == 'auto':
                self.line_plot_silhouette(n_clusters_eval, title=title, show=plot_fig, output=output_n_clusters)
        return best_n_clusters, best_clustering, best_silhouette_score, n_clusters_eval, token_records

    def clustering_raw_embedding(self, df, embed_column='embed', label_column='labels', n_cluster_range=[2,15,2],
                                 trials=30, plot_fig=False, dimen_reducetion_method=None,
                                 visualization_reduction_method='auto', output=None, possible_clusters='auto',
                                 token_name=''):
        """

        :param df:
        :param embed_column:
        :param label_column:
        :param n_cluster_range:
        :param trials:
        :param plot_fig:
        :param dimen_reducetion_method:
        :param visualization_reduction_method:
        :param output:
        :param possible_clusters:
        :param token_name: str; name of the token, used for title in visualization
        :return:
        """
        embeddings = [i[0] for i in df[embed_column]]
        best_silhouette_score = -1.1
        best_clustering = None
        best_n_clusters = None
        n_clusters_eval = []

        # try different values of n_clusters, for each n cluster perform kmeans clustering
        for n_clusters in range(n_cluster_range[0], n_cluster_range[1], n_cluster_range[2]):
            clustering = self.KMeans_clustering_bert_token(embeddings, n_clusters, trials=trials)
            # calculate average silhouette score for current clustering,
            # and record it for study of the trend of clustering
            average_silhouette_score = silhouette_score(embeddings, clustering.labels_)
            n_clusters_eval.append((n_clusters, average_silhouette_score))
            # record the best cluster
            if average_silhouette_score > best_silhouette_score:
                best_silhouette_score = average_silhouette_score
                best_n_clusters = n_clusters
                best_clustering = copy.deepcopy(clustering)
        # plot
        if plot_fig or output:
            title = "Kmeans clustering on " + token_name
            if visualization_reduction_method == 'auto' and dimen_reducetion_method is None:
                reduced = decomp.PCA(n_components=3).fit_transform(embeddings)
            if output == None:
                output = None
                if dimen_reducetion_method is not None:
                    title += ' with dimensional reduction method ' + dimen_reducetion_method
                if possible_clusters == 'auto':
                    output_n_clusters = None
            else:
                name = token_name + '_kmeans_n-clusters'+ str(best_n_clusters)
                if dimen_reducetion_method is not None:
                    name += '_dimen_reduction' + dimen_reducetion_method
                    title += ' with dimensional reduction method '+ dimen_reducetion_method
                if possible_clusters == 'auto':
                    name_n_clusters = name + '_n_clusters_silhouette.png'
                if visualization_reduction_method == 'auto' and dimen_reducetion_method is None:
                    name += '_visual_reductionPCA'
                name += '.png'
                output_n_clusters = os.path.join(output, name_n_clusters)
                output = os.path.join(output, name)
            self.silhouetteplot(embeddings, best_clustering.labels_, best_n_clusters, reduced, title=title,
                                show=plot_fig, output=output)
            if possible_clusters == 'auto':
                self.line_plot_silhouette(n_clusters_eval, title=title, show=plot_fig, output=output_n_clusters)
        return best_n_clusters, best_clustering, best_silhouette_score, n_clusters_eval

    def KMeans_clustering_bert_token(self, token_embeddings, n_clusters, trials=30, normalize=True):
        """
        This function takes a list of token embeddings, apply normalization on it and then apply kmeans clustering with
        some specific number of clusters on it.
        :param token_embeddings: list of np.ndarray, or a 2d np.ndarray. each row is an bert embedding
        :param n_clusters: number of clusters
        :param trials: int, number of times for k-means to run on different random initialized centroids.
        :param normalize: bool; if to normalize the token embedding before clustering
        :return: the clustering
        """
        if normalize:
            embeddings = preprocessing.normalize(token_embeddings)
        clustering = cluster.KMeans(n_clusters=n_clusters, n_init=trials, max_iter=500).fit(embeddings)
        return clustering

    def on_cluster_statistics(self, cluster_labels, embeds, dataset=None):
        """
        This function counts the line index and position index of tokens for each clusters
        The function will also check for the label if data set is provided
        The returning value of this method is a nested dictionary, each item in this dictionary represent some count
        value of this statistics, including a list of line index and a list of position index, if dataset is provided
        then it will also check the class label of the sentence that the token comes from. These statistics can be used
        to study how difference the distribution of token embeddings in the embedding space might be caused or might
        contribute to the classification process.
        :param cluster_labels: cluster labels obtained from clustering algorithms
        :param embeds: a list of tuple, each tuple contains embeddings lineindex and position index
        :param dataset: None or string; refers to the dataset to check
        :return: a nested dictionary
        """
        df = None
        if dataset:
            try:
                df = self.dfs[dataset]
            except KeyError:
                df = None
        result = {}
        for cluster in set(cluster_labels):
            result[cluster] = {'lineidx': [], 'positionidx': []}
            if df is not None:
                result[cluster]['label'] = []
        for i in range(len(cluster_labels)):
            result[cluster_labels[i]]['lineidx'].append(embeds[i][1])
            result[cluster_labels[i]]['positionidx'].append(embeds[i][2])
            if df is not None:
                line = embeds[i][1]
                result[cluster_labels[i]]['label'].append(df.iloc[line]['label'])
        return result

    def cluster_label_divergence(self, on_cluster_stat, divergence='jsd', parameter_dict={}, show=False,
                                 title=None, output=None):
        """
        Calculate group-wise divergence of the label, to see if different cluster of embedding suggests different
        distribution of class that they belongs to.
        After calculating the group-wise divergence, we use heatmap to illustrate the group-wise divergence. If show is
        set to True, then the plot is shown, if the output is not None, then the plot will be saved to target location.
        If output is None and show is False, then the plot will not be rendered at all
        :param on_cluster_stat: a nested python dictionary, generated using on_cluster_statistics
        :param divergence: string; indication which divergence calculation function to be used, could be either jsd
        for Jensen-Shannon divergence, skld for symmetric KL divergence. Any unidentified value will result in using jsd
        :param parameter_dict: python dictionary, optional parameters for divergence calculation method
        :param show: bool, if to show the plot.
        :param output: None or str
        :return: a square np.ndarray that has dimensionality of 2, array[i][j] = array[j][i] and represent the
        divergence of class label distribution between these two clusters
        """

        # cluster label starts from 0, so we add 1 to the max cluster number index to get the number of clusters
        num_clusters = np.max(list(on_cluster_stat.keys())) + 1
        # intialize the np.ndarray to -1, which is impossible for the divergence to achieve
        groupwise_div = np.full((num_clusters, num_clusters), -1.0)
        for i in range(num_clusters):
            for j in range(i, num_clusters):
                i_dist = {}
                j_dist = {}
                assert 'label' in set(on_cluster_stat[i].keys())
                assert 'label' in set(on_cluster_stat[j].keys())
                cluster_i_label = on_cluster_stat[i]['label']
                cluster_j_label = on_cluster_stat[j]['label']
                # get possible labels set for i and j th cluster
                possible_labels = set(cluster_i_label + cluster_j_label)
                # initialize the dictionary
                for label in possible_labels:
                    i_dist[label] = 0
                    j_dist[label] = 0
                # count the instance of label for each cluster
                for label in cluster_i_label:
                    i_dist[label] += 1
                for label in cluster_j_label:
                    j_dist[label] += 1
                # calculate divergence
                if divergence == 'skld':
                    try:
                        add_one_smooth = parameter_dict['add_one_smooth']
                    except KeyError:
                        add_one_smooth = True
                    div_ij = self.symmetric_kl_divergence(i_dist, j_dist, add_one_smooth)
                else:
                    try:
                        weighted = parameter_dict['weighted']
                    except KeyError:
                        weighted = False
                    div_ij = self.jensen_shannon_divergence(i_dist, j_dist, weighted)
                # fill the value in target location, we are only using symmetric divergence, so the result array will
                # also be symmetric
                groupwise_div[i][j] = div_ij
                groupwise_div[j][i] = div_ij
        #visualization
        if show or output is not None:
            ax = sns.heatmap(groupwise_div)
            # plt.imshow(groupwise_div, cmap='hot', interpolation='nearest', projection='2d')
            ax.set_title(title)
            if output is not None:
                plt.savefig(os.path.join(output, str(num_clusters)+'.png'))
            if show:
                plt.show()
            ax = None
            plt.close()

        return groupwise_div

    @staticmethod
    def rmv_punc(input_str):
        """
        Remove all punctuations from the string
        Return the new string
        :param input_str: str; input string
        :return: input string with punctuations removed
        """
        return ''.join(c for c in input_str if c not in punctuation)

    @staticmethod
    def symmetric_kl_divergence(P, Q, add_one_smooth=True):
        """
        Measure the symmetric kl divergence of distribution P and distribution Q
        Symmetric KL-divergence is measured as DKL(P || Q) + DKL(Q || P)
        Both distribution P and distribution Q should be a python dictionary object and each item is correspond to the
        number of instances of type key items in the correspond distribution.
        The keys() of these two dictionary should be equal and the full set of all possible item values.
        The value of the counts should be int. Could be 0 if the target type of instance does not exist in target
        distribution
        :param P: the first distribution
        :param Q: the second distribution
        :param add_one_smooth: Boolean, if to use add one smooth to add one instance of each type of item in both of the
        distribution to avoid zero division
        :return: the KL-divergence of distribution P and Q
        """
        assert P.keys() == Q.keys()
        # add one smoothing
        if add_one_smooth:
            for key in P.keys():
                P[key] += 1
                Q[key] += 1

        pq_diver = BERTEmbedManager.kl_divergence(P, Q)
        qp_diver = BERTEmbedManager.kl_divergence(Q, P)
        return pq_diver + qp_diver

    @staticmethod
    def kl_divergence(P, Q, normalized=False):
        """
        Calculate KL-divergence of two input distribution P and Q
        P and Q should be dictionary, each key represent a certain type of instances, and the value is the count or
        the probability of the occurrence of that instance.
        :param P:
        :param Q:
        :param normalized: if the P and Q are dictionary of count or probabilities
        :return:
        """
        # asymmetric version of KL-divergence
        p_values = []
        q_values = []
        for key in P.keys():
            if not Q[key] <= 0:
                p_values.append(P[key])
                q_values.append(Q[key])
        p_values = np.array(p_values).astype(np.float)
        q_values = np.array(q_values).astype(np.float)
        if normalized:
            p_total = 1.0
            q_total = 1.0
        else:
            p_total = np.sum(p_values)
            q_total = np.sum(q_values)
        diver = 0.0
        for i in range(p_values.shape[0]):
            if not p_values[i] == 0.0:
                diver += p_values[i] * np.log(p_values[i] * q_total / q_values[i] / p_total)


        diver /= p_total
        return diver

    @staticmethod
    def jensen_shannon_divergence(P, Q, weighted=False):
        """
        Calculate Jensen Shannon divergence based on the input distribution.
        Jensen Shannon distribution is a symmetric and smoothed version of KL-divergence
        The Jensen Shannon divergence is calculated in the following manner
        DJS = DKL(P || M) + DKL(Q || M)
        where DKL is the asymmetric version of KL divergence and M is the average of the two input distribution
        P and Q should be python dictionary, and have same keys() list. Each value of the item in the dictionary is the
        count of number of a certain type(represented by the key) of instances in the target distribution, should be
        an integer greater or equals to 1
        :param P: dictionary; the first distribution
        :param Q: dictionary; the second distribution
        :param weighted: bool; if to use weighted mean of two distribution to calculate M or even mean of the two
        distribution
        :return: float; measured Jensen-Shannon divergence among the two input distribution
        """
        # assert that the two set of distribution has the same key set
        assert P.keys() == Q.keys()
        M = {}
        if not weighted:
            # calculate the distribution of P and Q based on their count
            p_total = 0.0
            q_total = 0.0

            for key in P.keys():
                p_total += float(P[key])
                q_total += float(Q[key])
            # calculate distribution and average distribution of P and Q, which is M
            for key in P.keys():
                P[key] = float(P[key]) / p_total
                Q[key] = float(Q[key]) / q_total
                M[key] = (P[key] + Q[key]) / 2
            return BERTEmbedManager.kl_divergence(P, M, True) + BERTEmbedManager.kl_divergence(Q, M, True)
        else:
            for key in P.keys():
                M[key] = P[key] + Q[key]
            return BERTEmbedManager.kl_divergence(P, M, False) + BERTEmbedManager.kl_divergence(Q, M, False)
