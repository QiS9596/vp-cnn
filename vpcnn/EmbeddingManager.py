import copy
import random
import time
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sklearn.decomposition as decomp
from sklearn import cluster
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
Axes3D
import token_db
import tokenization


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
                word = self.db_manager.rmv_punc(word)
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
                    word = self.db_manager.rmv_punc(word)
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
                    word = self.db_manager.rmv_punc(word)
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
        ax.set_ylim([-100, len(data) + vertical_spacing])
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
            y_lower = y_upper + 10
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
        :param possible_clusters: str; if to plot the silhouette score for each
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
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1], n_clusters_range[2]):
            clustering = self.KMeans_clustering_bert_token(token_embeddings, n_clusters, trials=trials)
            average_silhouette_score = silhouette_score(token_embeddings, clustering.labels_)
            n_clusters_eval.append((n_clusters, average_silhouette_score))
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
                    title += ' with dimensional reduction method '+ dimen_reduction_method
                if possible_clusters == 'auto':
                    title_n_clusters = title + ' n-clusters silhouette'
                    output_n_clusters = None
            else:
                name = dataset_name+'_'+token_name+'_' + str(layer) + '_kmeans_n-clusters'+ str(best_n_clusters)
                if dimen_reduction_method is not None:
                    name += '_dimen-reduction' + dimen_reduction_method
                    title += ' with dimensional reduction method '+ dimen_reduction_method
                if possible_clusters == 'auto':
                    title_n_clusters = title + ' n-clusters silhouette'
                    name_n_clusters = name +'_n_clusters_silhouette.png'
                if visualize_reduction_method == 'auto' and dimen_reduction_method is None:
                    name += '_visual-reductionPCA'
                name += '.png'
                output_n_clusters = os.path.join(output, name_n_clusters)
                output = os.path.join(output, name)
            self.silhouetteplot(token_embeddings, best_clustering.labels_, best_n_clusters, reduced, title=title,
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
