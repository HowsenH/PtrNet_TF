'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from tqdm import tqdm

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def our_read_graph(distance_matrix, threshold):
	# TODO: Find unconnected edges, if distance() > 5, and set weight to 0
	# TODO: reverse the distance as weight 1/d
	weight = 1/distance_matrix
	weight[distance_matrix > threshold] = 0
	G = nx.from_numpy_matrix(weight, create_using=nx.DiGraph())
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)
	
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	x = np.load('../../data/distance_matrix5-20/n=20_index1_x.npy')
	num_graph, n, n = x.shape
	embeded_dim = 32
	embeded = np.zeros((num_graph, n, embeded_dim))
	print(x.shape)
	for i in tqdm(range(num_graph)):
		nx_G = our_read_graph(x[i], 10)
		# BFS style walk
		G = node2vec.Graph(nx_G, is_directed=True, p=10, q=1)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(num_walks=50, walk_length=20)
		# Word2Vec to embed walks
		walks = [map(str, walk) for walk in walks]
		model = Word2Vec(walks, size=32, window=args.window_size, min_count=0, sg=1, workers=args.workers,
						 iter=args.iter)
		for vocab in model.vocab:
			embeded[i, int(vocab),:] = model[vocab]
		# print(embeded[i])
		# print(y[i])
	np.save('../../data/distance_matrix5-20/embedded_n=20_index1_x.npy', embeded)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	# x = np.load('../../data/distance_matrix5-20/embedded_n=10_index1_x.npy')
