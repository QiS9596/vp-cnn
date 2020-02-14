import token_db
import numpy as np
import pandas as pd
import tokenization
import random
import time
from sklearn import cluster
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy


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

    def silhouetteplot(self, data, cluster_label, n_clusters):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([-1.0, 1.0])
        vertical_spacing = (n_clusters+1) * 10
        average = silhouette_score(data, cluster_label)
        sample_silhouette_values = silhouette_samples(data, cluster_label)

        y_lower = 10

        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_label==i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color_map = cm.get_cmap('nipy_spectral')
            color = color_map(float(i)/n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0,
                             ith_cluster_silhouette_values,
                             facecolor=color,
                             edgecolor=color,
                             alpha=0.7)
            ax.text(-0.05, y_lower+0.5*size_cluster_i, str(i))
            y_lower = y_upper +10
        ax.set_title('Silhouette Plot')
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Cluster Label')
        ax.axvline(x=average, color='red', linesytle='--')
        ax.set_yticks([])
        ax.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
        plt.show()


    def clustering_analysis_BERT_token(self, dataset_name, token_name, layer, n_clusters_range=[2,15,2], plot_fig=False):
        # collect the embeddings
        token_records = self.db_manager.get_embeddings_onLayer(dataset_name, token_name, layer)
        token_embeddings = []
        for token_record in token_records:
            token_embeddings.append(token_record[0])
        best_silhouette_score = -1.1
        best_clustering = None
        best_n_clusters = None
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1], n_clusters_range[2]):
            clustering = self.KMeans_clustering_bert_token(token_embeddings, n_clusters)
            average_silhouette_score = silhouette_score(token_embeddings, clustering.labels_)
            if average_silhouette_score > best_silhouette_score:
                best_silhouette_score = average_silhouette_score
                best_clustering = copy.deepcopy(clustering)
                best_n_clusters = n_clusters
        if plot_fig:
            self.silhouetteplot(token_embeddings, best_clustering.labels_, best_n_clusters)


    def KMeans_clustering_bert_token(self, token_embeddings, n_clusters):
        """
        This function takes a list of token embeddings, apply normalization on it and then apply kmeans clustering with
        some specific number of clusters on it.
        :param token_embeddings: list of np.ndarray, or a 2d np.ndarray. each row is an bert embedding
        :param n_clusters: number of clusters
        :return: 
        """
        embeddings = preprocessing.normalize(token_embeddings)
        clustering = cluster.KMeans(n_clusters=n_clusters, n_init=30, max_iter=500).fit(embeddings)
        return clustering
