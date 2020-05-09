import torch
import numpy as np 

def one_hot_encode(labels,num_labels=10):
	if labels.dim()==1:
		labels.unsqueeze_(1)
	
	labels_oh = torch.FloatTensor(labels.size(0),num_labels)
	labels_oh.zero_()
	labels_oh.scatter_(1,labels,1)

	return labels_oh

def get_random_labels(batch_size,num_labels=10):
	rand_labels = torch.randint(num_labels,(batch_size,1))
	return one_hot_encode(rand_labels)