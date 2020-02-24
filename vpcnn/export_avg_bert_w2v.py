import token_db
import argparse
import time
parser = argparse.ArgumentParser(description='Export Average BERT embeddings to word2vec txt file')
parser.add_argument('-output-dir', type=str, default='./data/w2v_avgBERT.txt')
parser.add_argument('-database', type=str, default='./data/bert_embed.db')
parser.add_argument('-dataset', type=str, default='VP1617STRICT')
parser.add_argument('-layer', type=str, default='-12')
args = parser.parse_args()
db = token_db.BERTDBProcessor(db_file=args.database)
list_token = db.get_list_tokens(args.dataset)
start = time.time()
count = 0
num_tk = len(list_token)
with open(args.output_dir, 'w') as file:
    for token in list_token:
        token = token
        embedding = db.get_avg_bert_embed(datasets_name=[args.dataset], token_name=token, layer=args.layer, line_selection=[None]).tolist()
        file.write(' '.join([token, ' '.join([str(i) for i in embedding])]))

        if count % 100 == 0:
            current = time.time()
            used = current-start
            estimate = used/(count+1)*(num_tk-count+1)
            print('{}/{} part of job done, used {}, estimated {}'.format(count, num_tk, used, estimate))
        count += 1
# db.c.execute("""
#     SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%colon%'
# """)
# print(db.c.fetchall())

# db.dump_table('VP1617STRICT_COLON')

