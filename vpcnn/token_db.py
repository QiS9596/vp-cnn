"""
This module provide tools for applying analysis towards bert embeddings
"""
import json
import sqlite3
import time
from string import punctuation

import numpy as np


class BERTDBProcessor:
    """
    This class holds the bert embedding for the entire dataset via sqlite and provide fundamental operations
    We read the dataset and the corresponded bert embeddings into this database, and save the bert embeddings at each
    token.
    The database is designed as following:
    TABLE BERT:
    a table that maintains a list of available dataset, in our initial design, should include 2 element:
    VP16 and VP1617Strict
    Each instance in this table contains one field text, refers to the name of the correspond dataset
    Note: for some BERT tokens that include some correspond punctuations like [] in [CLS], it will cause operational
    error in SQL command, we simply remove it in the token name
    -------------------------------------------------------------------------------------------------------------------
    TABLE dataset_name:
    a table that maintains a list of all token that appears in the dataset; the name of this table should be the name
    of the dataset, and each dataset_name TABLE should be related to an instance in TABLE BERT.
    we will have one table for each dataset.
    instance in this table:
    (token TEXT PRIMARY KEY, count INT)
    token refers to the text of the token
    count refers to the times that this token appears in the current dataset
    ------------------------------------------------------------------------------------------------------------------
    TABLE dataset_name_token:
    a table that maintains a list of embeddings of the current token that appears in the referred dataset. The name of
    this table should be the name of the dataset underscore the token. each of this kind of table should have a related
    instance in the related dataset table. we will have one table for each instance for each dataset.
    instance in this table:
    (layer TEXT, embedding TEXT, lineidx INT, positionidx INT)
    each instance correspond to a embedding of this token, a group of multiple embeddings (extracted from different
    layer and different method) correspond to an occurrence of this token in the correspond dataset
    layer: the layer that the BERT embeddings were extracted, for directly extracted feature, should be -1 to -12 in
    string; it could also be avg4 that refers to the average of last four layers.
    embedding: the BERT embedding that extract for this occurrence of this token in this dataset of that particular layer
    the text of this embedding is generated using np.dump()
    lineidx: the line index of this BERT embedding, which links this embedding to a training example in the dataset
    positionidx: the position index of this token in the correspond training example
    """

    def __init__(self, db_file='./data/bert_embed.db'):
        """
        link to target database file, create default BERT table
        :param db_file: path to database file
        """
        self.conn = sqlite3.connect(db_file)
        self.c = self.conn.cursor()
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS BERT (dataset TEXT PRIMARY KEY)
        """)
        self.conn.commit()

    def __del__(self):
        self.conn.close()

    def list_tables(self):
        """
        print a list of table in the entire dataset
        :return:
        """
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table'
        """)
        for table in self.c.fetchall():
            print(table)
        self.conn.commit()

    def create_dataset_table(self, table_name):
        """
        Create a dataset table, explained in detail above in section TABLE dataset_name
        :param table_name: name of the table
        :return:
        """
        self.c.execute("""
            INSERT OR IGNORE INTO BERT (dataset) values (?)
        """, [table_name])
        self.conn.commit()
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS {} (token TEXT PRIMARY KEY, count INT)
        """.format(table_name))
        self.conn.commit()

    def create_dataset_token_table(self, dataset_name, token_name):
        """
        Create a dataset-token table, which is described above as dataset_name_token
        This function will first insert a token data entry into target dataset table and create table that hold all the
        embeddings of this token in this dataset among all bert layers.
        The count field for this token in the correspond dataset table would be set to 0.
        Will not create table and return False if the target dataset table does not exist
        :param dataset_name: name of the correspond dataset
        :param token_name: name of the target token
        :return: False if target dataset table does not exist; otherwise True
        """
        # check if the correspond dataset table exist
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table' AND name=?;
        """, [dataset_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # if the correspond dataset table exists, insert the token entity into the dataset table
        self.c.execute("""
            INSERT OR IGNORE INTO {} (token, count) values (?,?)
        """.format(dataset_name), [token_name, 0])
        self.conn.commit()
        # after the insertion, create the correspond table
        table_name = dataset_name + '_' + token_name
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS {} (layer TEXT, embedding TEXT, lineidx INT, positionidx INT)
        """.format(table_name))
        self.conn.commit()
        return True

    def insert_embedding(self, dataset_name, token_name, layer, embedding, lineidx, positionidx):
        """
        Insert one embedding to target dataset_name_token table, if the target table does not exists, return False,
        else return True
        :param dataset_name: string; name of the target dataset name
        :param token_name: string; name of the target token
        :param layer: string or int; could be -1 to -12 for representing correspond bert layer, or could also be string
        like avg4 to represent processed embeddings as average of last four layers. The int query will be further
        converted to string format into the sql
        :param embedding: string or np.array; embedding for the target token extracted at the target location/method.
        if a string is passed, that string should come from np.array.dumps(); also if a np.array is passed, it will be
        processed with np.array.dumps() in this method
        :param lineidx: int; index of the sentence that this token belongs to
        :param position: int; index of token inside the sentence
        :return: False if the target table does not exists, True other wise
        """
        # check if the correspond dataset_name_token table exists
        table_name = dataset_name + '_' + token_name
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table' AND name=?;
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # insert the target embedding into the dataset_name_token table
        if isinstance(layer, int):
            layer = str(layer)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.dumps()
        self.c.execute("""
            INSERT INTO {} (layer, embedding, lineidx, positionidx) values (?,?,?,?)
        """.format(table_name), [layer, embedding, lineidx, positionidx])
        self.conn.commit()
        return True

    def get_embeddings(self, dataset_name, token_name):
        """
        Get a collection of word embedding for all appearances of target token in the correspond dataset
        Return None if the target dataset_name_token table does not exist. A list of string otherwise. The string is
        generated use np.array.dumps() method and could be used to reconstruct np.array with np.loads()
        :param dataset_name: string; name of the correspond dataset
        :param token_name: string; name of the target token
        :return: None if the table of target dataset_name_token table does not exist; otherwise a list
        """
        table_name = dataset_name + '_' + token_name
        # check if the target table exists
        self.c.execute("""
            SELECT  * FROM sqlite_master WHERE type='table' AND name=?;
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # get the collection of token embeddings
        self.c.execute("""
            SELECT embedding FROM {}
        """.format(table_name))
        result = self.c.fetchall()
        self.conn.commit()
        return result

    def dump_table(self, table_name):
        """
        This function will print every data entry in the target table with the name table_name
        The function will first check if the target table exists, if not it terminates
        if the target table exists, it'll print the data in target table
        :param table_name: string; name of the target table
        :return: False if target table does not exist, True if target table exist
        """
        self.c.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name=?;
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) > 0:
            self.c.execute("""
                SELECT * FROM {}
                """.format(table_name))
            print(self.c.fetchall())
            self.conn.commit()
            return True
        return False

    def drop_tables(self):
        """
        This function drop all of the tables exists in the dataset
        :return:
        """
        self.c.execute("""
            SELECT name FROM sqlite_master WHERE type='table';
        """)
        table_names = self.c.fetchall()
        self.conn.commit()

        for table in table_names:
            self.c.execute("""
                DROP TABLE {}
            """.format(table[0]))
            self.conn.commit()

    def increment_token_count(self, dataset_name, token_name, increment_amount=1):
        """
        Increment the count of target token in target dataset table
        :param dataset_name: name of the target dataset
        :param token_name: name of the target token
        :param increment_amount: int; the amount that increments upon the current count
        :return: False if target dataset or token does not exist; True otherwise
        """
        # we directly check the correspond dataset_name_token table
        table_name = dataset_name + '_' + token_name
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table' and name=?
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0: return False

        # get the current count
        self.c.execute("""
            SELECT count FROM {} WHERE token=?
        """.format(dataset_name), [token_name])
        count = self.c.fetchone()[0]
        # store the new_count into target location
        new_count = count + increment_amount
        self.c.execute("""
            UPDATE {} SET count=? WHERE token=?
        """.format(dataset_name), [new_count, token_name])
        self.conn.commit()
        return True

    def load_dataset(self, bert_json_file, dataset_name):
        """
        This function load the target bert json file into the target dataset table/table with target dataset prefix
        This function will first check if the correspond table exists, if not, will create correspond table
        :param bert_json_file: path to target bert json file
        :return:
        """
        print('start to load dataset')
        self.create_dataset_table(dataset_name)
        progress = 0
        file_lines = self.getline(bert_json_file)
        with open(bert_json_file) as file:
            start = time.time()
            print(start)
            for line in file.readlines():
                line_dict = json.loads(line)
                lineidx = line_dict['linex_index']
                features = line_dict['features']
                for i in range(len(features)):
                    token = features[i]
                    token_name = self.rmv_punc(token['token'])
                    self.create_dataset_token_table(dataset_name, token_name)
                    self.increment_token_count(dataset_name, token_name)
                    for layer in token['layers']:
                        layeridx = layer['index']
                        embed = np.array(layer['values'])
                        self.insert_embedding(dataset_name, token_name,
                                              layer=layeridx, embedding=embed, lineidx=lineidx, positionidx=i)
                if progress % 100 == 0:
                    unfinshed = file_lines - progress
                    time_used = time.time() - start
                    estimated_finish = time_used / (progress+1) * unfinshed
                    print("Current Progress {}/{}, has already work on job for {}, estimated end in {} sec".format(
                        progress, file_lines, time_used, estimated_finish))
                progress += 1

    @staticmethod
    def getline(file):
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def rmv_punc(string_):
        """
        remove the punctuations for the input string
        :param string_: input string
        :return: string_ with punctuations removed
        """
        return ''.join(c for c in string_ if c not in punctuation)
