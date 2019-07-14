    
import argparse
from random_forest import Random_forest
parser = argparse.ArgumentParser(description='3SRF')
parser.add_argument('--n-trees', default=10, type=int, help='number of decision tree')
parser.add_argument('--n-processes', default=1, type=int, help='number of processes')
parser.add_argument('--n-max-features', default=5, type=int, help='number of features')
parser.add_argument('--max-depth', default=5, type=int, help='max depth for decision tree')
parser.add_argument('--min-sample-split', default=5, type=int, help='minimun number of sample to split')
args = parser.parse_args()
# n_trees=10, n_processes=1, num_max_features=5, max_depth=5, min_samples_split=5

train_files = [f'./data/train{x}' for x in range(1,6)]

test_files = [f'./data/test{x}' for x in range(1,6)]

label_files = [f'./data/label{x}' for x in range(1,6)]

model = Random_forest(n_trees=args.n_trees, n_processes=args.n_processes, num_max_features=args.n_max_features, max_depth=args.max_depth, min_samples_split=args.min_sample_split)


model.fit(train_files,label_files)

result = model.predict(test_files)

pd.DataFrame(result,columns=['Predicted'],index=list(range(1,len(result) + 1))).to_csv('./data/result.csv')