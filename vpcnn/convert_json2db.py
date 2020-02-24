import token_db
import argparse

parser = argparse.ArgumentParser("Convert bert json file into sqlite database")
parser.add_argument('-db', type=str, default='./data/bert_embed.db', help='database file for storage [default:./data/bert_embed.db]')
parser.add_argument('-dataset', type=str, default='VP1617STRICT', help='dataset name [default:VP1617STRICT]')
parser.add_argument('-jsonpath', type=str, default='./data/bert_embeddings_json_new/train_feature.json', help='path to bert json file')
args = parser.parse_args()
db = token_db.BERTDBProcessor(db_file=args.db)
db.create_dataset_table(args.dataset)
print('dataset table created')
db.load_dataset(args.jsonpath, args.dataset)
