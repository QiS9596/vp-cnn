import token_db
import numpy as np
import pandas as pd
import tokenization
import random
import time


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
        # in the validation set, we build the embedding based on the dictionary we generated in the previous step
        # if OOV is found, simply randomly choose any one of the word in the dictionary as a replacement
        dev_result = []
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
                        random_word = random.choice(embed_dict.keys())
                        sentence_embed.append(embed_dict[random_word])
                dev_result.append(sentence_embed)
        return (embed_dict, train_result, dev_result)
