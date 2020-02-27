"""
This module provide tools for applying analysis towards bert embeddings
"""
import json
import sqlite3
import time

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
        self.punc = {'[': 'LEFT_BRACKET', ']': 'RIGHT_BRACKET', '(': 'LEFT_PARENTHESIS', ')': 'RIGHT_PARENTHESIS',
                     '.': 'DOT', ',': 'COMMA', '#': 'SHARP', '!': 'EXCLAM', '-': 'DASH', '?': 'QUES', '\'': 'APOS',
                     '\"': 'QUOT', '*': 'ASTE', '/':'SLASH', '\\':'RETURN',';':'SCOLON', ':':'COLONSIGN', '>':'GT', '<':'LT',
                     '&':'ANDSIGN'}

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
        token_name = self.rpl_punc(token_name)
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
        try:
            self.c.execute("""
                CREATE TABLE IF NOT EXISTS {} (layer TEXT, embedding TEXT, lineidx INT, positionidx INT)
            """.format(table_name))
            self.conn.commit()
        except sqlite3.OperationalError:
            print("err" + table_name)
            raise KeyboardInterrupt
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
        token_name = self.rpl_punc(token_name)
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

    def get_embeddings_onLayer(self, dataset_name, token_name, layer):
        """
        Get bert embedding of a certain token in target dataset that are only on certain layer
        :param dataset_name: name of the dataset
        :param token_name: name of the token
        :param layer: layer indicator, str or int
        :return: False or list of np.array
        """
        table_name = dataset_name + '_' + token_name
        # check if the target table exists
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table' AND name=?
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # get the collection of token embeddings
        if not isinstance(layer, str):
            layer=str(layer)
        self.c.execute("""
            SELECT embedding, lineidx, positionidx FROM {} WHERE layer=?
        """.format(table_name), [layer])
        result = self.c.fetchall()
        self.conn.commit()
        return result

    def get_list_tokens(self, dataset_name):
        """
        get a list of tokens for the particular dataset
        :param dataset_name: name of the target dataset
        :return: list of str, each one is an embedding
        """
        # check if target dataset exists
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type='table' AND name=?
        """, [dataset_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # then fetch the tokens for this dataset
        self.c.execute("""
            SELECT token FROM {} 
        """.format(dataset_name))
        result = self.c.fetchall()
        self.conn.commit()
        result_ = [i[0] for i in result]
        return result_


    def get_embeddings(self, dataset_name, token_name):
        """
        Get a collection of word embedding for all appearances of target token in the correspond dataset
        Return None if the target dataset_name_token table does not exist. A list of string otherwise. The string is
        generated use np.array.dumps() method and could be used to reconstruct np.array with np.loads()
        :param dataset_name: string; name of the correspond dataset
        :param token_name: string; name of the target token
        :return: False if the table of target dataset_name_token table does not exist; otherwise a list
        """
        token_name = self.rpl_punc(token_name)
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
        return self.unpack_fetch_all_embeddings(result)

    def get_embeddings_on_row(self, dataset_name, token_name, line_ids, layer):
        """
        get a collection of word embedding for selected appearances of target token
        the token is selected by lineidx field, which is the index of the example in the training set
        :param dataset_name: name of the dataset
        :param token_name: name of the token
        :param line_ids: a list of integer, each one represents a line idl; could also be None, in this case return
        search among entire dataset
        :param layer: str or int, layer for embedding; should correspond to -1 to -12 and other method of extracting
        embeddings
        :return: False if the table of target dataset_name_token does not exist; otherwise a list of numpy.array
        """
        layer = str(layer)
        token_name = self.rpl_punc(token_name)
        table_name = dataset_name + '_' + token_name
        # check if the target table exists
        self.c.execute("""
            SELECT * FROM sqlite_master WHERE type ='table' AND name=?;
        """, [table_name])
        self.conn.commit()
        if len(self.c.fetchall()) <= 0:
            return False
        # if train_line_ids is None
        if line_ids is None:
            self.c.execute("""
                SELECT embedding FROM {} WHERE layer=?
            """.format(table_name), [layer])
        else:
            str_list_sql = '('
            for i in range(len(line_ids)):
                str_list_sql += str(line_ids[i])
                if i < len(line_ids) - 1:
                    str_list_sql += ','
            str_list_sql += ')'
            # get the collection of token embeddings
            self.c.execute("""
                SELECT embedding FROM {} WHERE lineidx in {} AND layer=?
            """.format(table_name, str_list_sql), [layer])
        result = self.c.fetchall()
        self.conn.commit()
        return self.unpack_fetch_all_embeddings(result)

    def get_avg_bert_embed(self, datasets_name, token_name, line_selection=None, layer=-1):
        """
        get the average bert embedding for target token across a collection of datasets, constrained to appearances in certain
        lines
        :param datasets_name: list of name of the datasets
        :param token_name: name of the token
        :param line_selection: a list contains lists of integer or None; if list each one represents a line id;
        if None include the entire dataset.
        :param layer: str or int, layer for embedding; should correspond to -1 to -12 and other method of extracting
        embeddings
        :return: False if target table does not exists, otherwise a numpy array as the average embedding of target token
        """
        token_name = self.rpl_punc(token_name)
        embeds_ = []
        for i in range(len(datasets_name)):
            dataset_name = datasets_name[i]
            line_ids = line_selection[i]
            embeds = self.get_embeddings_on_row(dataset_name, token_name, line_ids, layer)
            if embeds:
                embeds_ += embeds
        result = np.average(embeds_, axis=0)
        if np.isnan(result).any():
            print(token_name)
            print(embeds_)
        return result

    @staticmethod
    def unpack_fetch_all_embeddings(embeds):
        """
        The database system will return a list of tuples as the result of select and getchall command
        when selecting an embedding. each tuple contains one embedding, in this case, a string that was created by
        np.array.dumps
        This method turns the unpacked embeddings as list of numpy.array
        :param embeds: database select result
        :return: list of numpy.array
        """
        result = []
        for embed in embeds:
            result.append(np.loads(embed[0]))
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
        token_name = self.rpl_punc(token_name)
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
                    token_name = self.rpl_punc(token['token'])

                    # token_name = token['token']
                    a = self.create_dataset_token_table(dataset_name, token_name)
                    b = self.increment_token_count(dataset_name, token_name)
                    if token['token'] == '&':
                        print('found token &')
                        print(a)
                        print(b)
                    for layer in token['layers']:
                        layeridx = layer['index']
                        embed = np.array(layer['values'])
                        self.insert_embedding(dataset_name, token_name,
                                              layer=layeridx, embedding=embed, lineidx=lineidx, positionidx=i)
                if progress % 100 == 0:
                    unfinshed = file_lines - progress
                    time_used = time.time() - start
                    estimated_finish = time_used / (progress + 1) * unfinshed
                    print("Current Progress {}/{}, has already work on job for {}, estimated end in {} sec".format(
                        progress, file_lines, time_used, estimated_finish))
                progress += 1

    def test_fun(self):
        self.c.execute("""
            SELECT * FROM VP1617STRICT
        """)
        result = self.c.fetchall()
        self.conn.commit()
        return result

    @staticmethod
    def getline(file):
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def rpl_punc(self, string_):
        """
        replace the punctuations for the input string to avoid syntax conflict with db
        :param string_: input string
        :return: string_ with punctuations removed
        """

        replaced = ''.join(c if c not in self.punc.keys() else self.punc[c] for c in string_)
        # print(replaced)
        return replaced
