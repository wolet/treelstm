"""
Preprocessing script to generate generate subtree data

"""
import sys
import os
import glob
from collections import defaultdict

#
# Trees and tree loading
#

class ConstTree(object):
	def __init__(self):
		self.left = None
		self.right = None
		self.label = -10

	def size(self):
		self.size = 1
		if self.left is not None:
			self.size += self.left.size()
		if self.right is not None:
			self.size += self.right.size()
		return self.size

	def set_spans(self):
		if self.word is not None:
			self.span = self.word
			return self.span

		self.span = self.left.set_spans()
		if self.right is not None:
			self.span += ' ' + self.right.set_spans()
		return self.span

	def get_labels(self, spans, labels, dictionary):
		if self.span in dictionary:
			spans[self.idx] = self.span
			labels[self.idx] = dictionary[self.span]
			self.label = dictionary[self.span]
		if self.left is not None:
			self.left.get_labels(spans, labels, dictionary)
		if self.right is not None:
			self.right.get_labels(spans, labels, dictionary)

	def print_tree(self, output_tuple, idx, root_only = False):
		if self.left is None and self.right is None:
			return output_tuple

		self.set_idx(0)
		self.set_parent(0)
		p,l = self.get_parents()
		output_tuple += [(self.size,self.span,p,l,str(idx)+'.'+str(self.identity))]

		if root_only:
			return output_tuple

		if self.left is not None:
			output_tuple = self.left.print_tree(output_tuple, idx)
		if self.right is not None:
			output_tuple = self.right.print_tree(output_tuple, idx)
		return output_tuple

	def find_leaf_parent(self, p_list, l_list):
		if self.left is None and self.right is None:
			return p_list + [self.parent], l_list + [self.label]

		if self.left is not None:
			p_list, l_list = self.left.find_leaf_parent(p_list, l_list)
		if self.right is not None:
			p_list, l_list = self.right.find_leaf_parent(p_list, l_list)
		return p_list, l_list

	def find_inner_parents(self, p_list, l_list):
		if self.left is None and self.right is None:
			return p_list, l_list
		if self.left is not None:
			p_list, l_list = self.left.find_inner_parents(p_list, l_list)
		if self.right is not None:
			p_list, l_list = self.right.find_inner_parents(p_list, l_list)
		return p_list + [self.parent], l_list + [self.label]

	def get_parents(self):
		tokens = self.span.split(' ')
		p_list,l_list = self.find_leaf_parent([],[])
		p_list,l_list = self.find_inner_parents(p_list,l_list)
		return p_list,l_list

	def set_parent(self, parent_idx):
		self.parent = parent_idx
		if self.left is not None:
			self.left.set_parent(self.idx)
		if self.right is not None:
			self.right.set_parent(self.idx)

	def set_leaf_idx(self, idx):

		if self.left is not None:
			idx = self.left.set_leaf_idx(idx)
		if self.right is not None:
			idx = self.right.set_leaf_idx(idx)

		if self.left is None and self.right is None:
			idx += 1
			self.idx = idx
		return idx

	def set_inner_idx(self, idx):

		if self.left is not None:
			idx = self.left.set_inner_idx(idx)
		if self.right is not None:
			idx = self.right.set_inner_idx(idx)

		if self.left is None and self.right is None:
			return idx

		idx += 1
		self.idx = idx
		return idx

	def set_idx(self,idx):
		idx = self.set_leaf_idx(idx)
		self.set_inner_idx(idx)

	def set_leaf_identity(self, idx):

		if self.left is not None:
			idx = self.left.set_leaf_identity(idx)
		if self.right is not None:
			idx = self.right.set_leaf_identity(idx)

		if self.left is None and self.right is None:
			idx += 1
			self.identity = idx
		return idx

	def set_inner_identity(self, idx):

		if self.left is not None:
			idx = self.left.set_inner_identity(idx)
		if self.right is not None:
			idx = self.right.set_inner_identity(idx)

		if self.left is None and self.right is None:
			return idx

		idx += 1
		self.identity = idx
		return idx

	def set_identity(self,idx):
		idx = self.set_leaf_identity(idx)
		self.set_inner_identity(idx)



def load_trees(dirpath):
	const_trees, toks = [], []
	with open(os.path.join(dirpath, 'parents.txt')) as parentsfile, \
		 open(os.path.join(dirpath, 'sents.txt')) as toksfile:
		parents = []
		for line in parentsfile:
			parents.append(map(int, line.split()))
		for line in toksfile:
			toks.append(line.strip().split())
		for i in xrange(len(toks)):
			const_trees.append(load_constituency_tree(parents[i], toks[i]))
	return const_trees, toks

def load_constituency_tree(parents, words):
	trees = []
	root = None
	size = len(parents)
	for i in xrange(size):
		trees.append(None)

	word_idx = 0
	for i in xrange(size):
		if not trees[i]:
			idx = i
			prev = None
			prev_idx = None
			word = words[word_idx]
			word_idx += 1
			while True:
				tree = ConstTree()
				parent = parents[idx] - 1
				tree.word, tree.parent, tree.idx = word, parent, idx
				word = None
				if prev is not None:
					if tree.left is None:
						tree.left = prev
					else:
						tree.right = prev
				trees[idx] = tree
				if parent >= 0 and trees[parent] is not None:
					if trees[parent].left is None:
						trees[parent].left = tree
					else:
						trees[parent].right = tree
					break
				elif parent == -1:
					root = tree
					break
				else:
					prev = tree
					prev_idx = idx
					idx = parent
	return root

def load_sents(dirpath, fname = 'sents.txt'):
	sents = []
	with open(os.path.join(dirpath, fname)) as sentsfile:
		for line in sentsfile:
			sent = ' '.join(line.split(' '))
			sents.append(sent.strip())
	return sents

def load_parents(dirpath, fname = 'parents.txt'):
	parents = []
	with open(os.path.join(dirpath, fname)) as parentsfile:
		for line in parentsfile:
			p = ' '.join(line.split(' '))
			parents.append(p.strip())
	return parents

def load_dictionary(dirpath):
	labels = []
	with open(os.path.join(dirpath, 'sentiment_labels.txt')) as labelsfile:
		labelsfile.readline()
		for line in labelsfile:
			idx, rating = line.split('|')
			idx = int(idx)
			rating = float(rating)
			if rating <= 0.2:
				label = -2
			elif rating <= 0.4:
				label = -1
			elif rating > 0.8:
				label = +2
			elif rating > 0.6:
				label = +1
			else:
				label = 0
			labels.append(label)
	d = {}
	with open(os.path.join(dirpath, 'dictionary.txt')) as dictionary:
		for line in dictionary:
			s, idx = line.split('|')
			d[s] = labels[int(idx)]
	return d

def get_labels(tree, dictionary):
	size = tree.size()
	spans, labels = [], []
	for i in xrange(size):
		labels.append(None)
		spans.append(None)
	tree.get_labels(spans, labels, dictionary)
	return spans, labels

def set_labels(dirpath, dictionary):
	const_trees, toks = load_trees(dirpath)
	for i in xrange(len(const_trees)):
		const_trees[i].set_spans()
		s, l = [], []
		for j in xrange(const_trees[i].size()):
			s.append(None)
			l.append(None)
		const_trees[i].get_labels(s, l, dictionary)
	return const_trees
if __name__ == '__main__':

	dictionary_path = sys.argv[1]
	label_path = sys.argv[2]
	output_path = sys.argv[3]
	threshold = int(sys.argv[4])
	root_only = False
	if len(sys.argv) > 5:
		root_only = True
		print "ROOT ONLY!"

	d = load_dictionary(dictionary_path)
	t = set_labels(label_path, d)
	output_tuple = []
	for idx, tree in enumerate(t):
		tree.set_identity(0)
		output_tuple = tree.print_tree(output_tuple,idx, root_only)

	stats = defaultdict(int)
	for tpl in output_tuple:
		size,span,p,l,idx = tpl
#		print size, span, '|', ' '.join(map(str, p)), '|' ,' '.join(map(str, l))
		stats[(size + 1)/2] += 1

	prev_cnt = 0
	prev_idx = 0
	totl_cnt = 0

	bucket_idx={}
	for l in stats:
		totl_cnt += stats[l]
		if prev_cnt == 0:
			prev_idx += 1
		prev_cnt += stats[l]
		if prev_cnt >= threshold:
			prev_cnt = 0
		bucket_idx[l] = prev_idx
		print l, prev_idx, stats[l], prev_cnt
	print "total", totl_cnt
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	out_f_sents = []
	out_f_label = []
	out_f_parnt = []
	out_f_idx = []

	for i in xrange(prev_idx):
		out_f_sents += [ open(os.path.join(output_path, 'sents.txt.' + str(i) ), 'w')]
		out_f_label += [ open(os.path.join(output_path, 'labels.txt.' + str(i) ), 'w')]
		out_f_parnt += [ open(os.path.join(output_path, 'parents.txt.' + str(i) ), 'w')]
		out_f_idx   += [ open(os.path.join(output_path, 'idx.txt.' + str(i) ), 'w')]

	for tpl in output_tuple:
		size,span,p,l,idx = tpl
		bucket = bucket_idx[(size+1)/2] -1
		out_f_sents[bucket].write(span+'\n')
		out_f_label[bucket].write(' '.join(map(str, l))+'\n')
		out_f_parnt[bucket].write(' '.join(map(str, p))+'\n')
		out_f_idx[bucket].write(idx+'\n')

	for i in xrange(prev_idx):
		out_f_sents[i].close()
		out_f_label[i].close()
		out_f_parnt[i].close()
		out_f_idx[i].close()
