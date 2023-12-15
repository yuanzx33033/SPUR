import os

os.environ["DATA_FOLDER"] = "./"


import argparse
import time

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()
import random

from utils.parser import *
from utils import datasets

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from sklearn.metrics import average_precision_score

from SPL_gp import SPLLoss_CL
from modules.gin.utils import get_adjacency_matrix
from modules.gin.GIN import GraphCNN

import warnings
warnings.filterwarnings("ignore")


def compute_kl_loss(p, q, pad_mask=None):
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

	# pad_mask is for seq-level tasks
	if pad_mask is not None:
		p_loss.masked_fill_(pad_mask, 0.)
		q_loss.masked_fill_(pad_mask, 0.)

	# You can choose whether to use function "sum" and "mean" depending on your task
	p_loss = p_loss.sum()
	q_loss = q_loss.sum()

	loss = (p_loss + q_loss) / 2
	return loss


def get_constr_out(x, R):
	c_out = x.double()

	c_out = c_out.unsqueeze(1)

	c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
	R_batch = R.expand(len(x), R.shape[1], R.shape[1])
	final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
	return final_out


class ConstrainedFFNNModel(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, hierarchy_size,
				 device, Nh):
		super(ConstrainedFFNNModel, self).__init__()

		self.nb_layers = hyperparams['num_layers']
		self.R = R
		self.hierarchy_size = np.cumsum(hierarchy_size).tolist()
		self.output_dim = output_dim

		fc = []
		for i in range(self.nb_layers):
			if i == 0:
				fc.append(nn.Linear(input_dim, hidden_dim))
			elif i == 1:
				fc.append(nn.Linear(2*hidden_dim, Nh*hidden_dim))
			elif i == self.nb_layers - 1:
				fc.append(nn.Linear(Nh*hidden_dim, output_dim))
			else:
				fc.append(nn.Linear(Nh*hidden_dim, Nh*hidden_dim))
		self.fc = nn.ModuleList(fc)

		self.drop = nn.Dropout(hyperparams['dropout'])

		self.sigmoid = nn.Sigmoid()
		if hyperparams['non_lin'] == 'tanh':
			self.f = nn.Tanh()
		else:
			self.f = nn.ReLU()

		self.device = device

		self.batch_norm = nn.BatchNorm1d(hidden_dim)

	# self.fc_last = nn.Linear(output_dim*2,output_dim)

	def forward(self, x, node_feats, adjacency_matrix=None):
		for i in range(self.nb_layers):
			if i == self.nb_layers - 1:

				x = self.fc[i](x)
				x = self.sigmoid(x)

			elif i == 0:
				node_feats = node_feats.repeat(x.size(0), 1)
				x = self.f(self.fc[i](x))
				x = torch.cat([x, node_feats], dim=1)

				x = self.drop(x)

			else:
				x = self.f(self.fc[i](x))
				x = self.drop(x)

		if self.training:
			masks = []
			# masking different levels of output given hierarchy h_i
			for h_i in range(len(self.hierarchy_size)):
				mask = torch.zeros_like(x)
				mask[:, :self.hierarchy_size[h_i]] = 1
				masks.append(mask)
			return x, None, masks
		else:
			constrained_out = get_constr_out(x, self.R)
			return constrained_out


def main():
	parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

	# Required  parameter
	parser.add_argument('--dataset', type=str, default=None, required=True,
						help='dataset name, must end with: "_GO", "_FUN", or "_others"')
	# Other parameters
	parser.add_argument('--seed', type=int, default=0,
						help='random seed (default: 0)')
	parser.add_argument('--device', type=str, default='0',
						help='GPU (default:0)')
	args = parser.parse_args()

	assert ('_' in args.dataset)
	assert ('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

	# Load train, val and test set
	dataset_name = args.dataset
	data = dataset_name.split('_')[0]
	ontology = dataset_name.split('_')[1]

	# Dictionaries with number of features and number of labels for each dataset
	input_dims = {'diatoms': 371, 'enron': 1001, 'imclef07a': 80, 'imclef07d': 80, 'cellcycle': 77, 'derisi': 63,
				  'eisen': 79, 'expr': 561, 'gasch1': 173, 'gasch2': 52, 'seq': 529, 'spo': 86}
	output_dims_FUN = {'cellcycle': 499, 'derisi': 499, 'eisen': 461, 'expr': 499, 'gasch1': 499, 'gasch2': 499,
					   'seq': 499, 'spo': 499}
	output_dims_GO = {'cellcycle': 4122, 'derisi': 4116, 'eisen': 3570, 'expr': 4128, 'gasch1': 4122, 'gasch2': 4128,
					  'seq': 4130, 'spo': 4116}
	output_dims_others = {'diatoms': 398, 'enron': 56, 'imclef07a': 96, 'imclef07d': 46, 'reuters': 102}
	output_dims = {'FUN': output_dims_FUN, 'GO': output_dims_GO, 'others': output_dims_others}

	# Dictionaries with the hyperparameters associated to each dataset
	num_gnn_layers_FUN = {'cellcycle': 3, 'derisi': 3, 'eisen': 3, 'expr': 1, 'gasch1': 2, 'gasch2': 2,
					   'seq': 1, 'spo': 1}
	num_gnn_layers_GO = {'cellcycle': 3, 'derisi': 1, 'eisen': 3, 'expr': 1, 'gasch1': 2, 'gasch2': 3,
					  'seq': 1, 'spo': 1}
	num_gnn_layers_others = {'diatoms': 2, 'enron': 2, 'imclef07a': 2, 'imclef07d': 2}
	num_gnn_layers = {'FUN': num_gnn_layers_FUN, 'GO': num_gnn_layers_GO, 'others': num_gnn_layers_others}

	num_mlp_layers_FUN = {'cellcycle': 1, 'derisi': 1, 'eisen': 1, 'expr': 1, 'gasch1': 1, 'gasch2': 1,
					   'seq': 2, 'spo': 2}
	num_mlp_layers_GO = {'cellcycle': 1, 'derisi': 1, 'eisen': 1, 'expr': 1, 'gasch1': 2, 'gasch2': 1,
					  'seq': 1, 'spo': 1}
	num_mlp_layers_others = {'diatoms': 2, 'enron': 2, 'imclef07a': 2, 'imclef07d': 2}
	num_mlp_layers = {'FUN': num_mlp_layers_FUN, 'GO': num_mlp_layers_GO, 'others': num_mlp_layers_others}

	graph_pooling_type_FUN = {'cellcycle': 'average', 'derisi': 'average', 'eisen': 'average', 'expr': 'sum', 'gasch1': 'average', 'gasch2': 'average',
						  'seq': 'sum', 'spo': 'average'}
	graph_pooling_type_GO = {'cellcycle': 'average', 'derisi': 'average', 'eisen': 'average', 'expr': 'sum', 'gasch1': 'average', 'gasch2': 'average',
						 'seq': 'sum', 'spo': 'average'}
	graph_pooling_type_others = {'diatoms': 'average', 'enron': 'average', 'imclef07a': 'average', 'imclef07d': 'average'}
	graph_pooling_type = {'FUN': graph_pooling_type_FUN, 'GO': graph_pooling_type_GO, 'others': graph_pooling_type_others}

	neighbor_pooling_type_FUN = {'cellcycle': 'average', 'derisi': 'average', 'eisen': 'average', 'expr': 'sum',
							  'gasch1': 'average', 'gasch2': 'average',
							  'seq': 'sum', 'spo': 'average'}
	neighbor_pooling_type_GO = {'cellcycle': 'average', 'derisi': 'average', 'eisen': 'average', 'expr': 'sum',
							 'gasch1': 'average', 'gasch2': 'average',
							 'seq': 'sum', 'spo': 'average'}
	neighbor_pooling_type_others = {'diatoms': 'average', 'enron': 'average', 'imclef07a': 'average',
								 'imclef07d': 'average'}
	neighbor_pooling_type = {'FUN': neighbor_pooling_type_FUN, 'GO': neighbor_pooling_type_GO,
						  'others': neighbor_pooling_type_others}

	hidden_dims_FUN = {'cellcycle': 500, 'derisi': 500, 'eisen': 600, 'expr': 2000, 'gasch1': 500, 'gasch2': 500,
					   'seq': 3000, 'spo': 500}
	hidden_dims_GO = {'cellcycle': 1000, 'derisi': 1000, 'eisen': 600, 'expr': 2000, 'gasch1': 500, 'gasch2': 500,
					  'seq': 12000, 'spo': 500}
	hidden_dims_others = {'diatoms': 3000, 'enron': 4000, 'imclef07a': 2000, 'imclef07d': 3000}

	hidden_dims = {'FUN': hidden_dims_FUN, 'GO': hidden_dims_GO, 'others': hidden_dims_others}

	learning_pace_w_FUN = {'cellcycle': 0.1, 'derisi': 0, 'eisen': 0, 'expr': 0, 'gasch1': 0, 'gasch2': 0,
						   'seq': 0, 'spo': 0}
	learning_pace_w_GO = {'cellcycle': 0, 'derisi': 0, 'eisen': 0, 'expr': 0, 'gasch1': 0, 'gasch2': 0,
						  'seq': 0, 'spo': 0}
	learning_pace_w_others = {'diatoms': 0, 'enron': 0, 'imclef07a': 0, 'imclef07d': 0.1}

	learning_pace_w = {'FUN': learning_pace_w_FUN, 'GO': learning_pace_w_GO, 'others': learning_pace_w_others}

	threshold_update_freq_FUN = {'cellcycle': 1, 'derisi': 8, 'eisen': 20, 'expr': 20, 'gasch1': 8, 'gasch2': 8,
								 'seq': 3, 'spo': 20}
	threshold_update_freq_GO = {'cellcycle': 1, 'derisi': 20, 'eisen': 20, 'expr': 20, 'gasch1': 8, 'gasch2': 8,
								'seq': 2, 'spo': 5}
	threshold_update_freq_others = {'diatoms': 800, 'enron': 20, 'imclef07a': 400, 'imclef07d': 100}

	threshold_update_freq = {'FUN': threshold_update_freq_FUN, 'GO': threshold_update_freq_GO, 'others': threshold_update_freq_others}

	ratio_beta_FUN = {'cellcycle': 0.1, 'derisi': 0.1, 'eisen': 0.6, 'expr': 0.6, 'gasch1': 0.3, 'gasch2': 0.6,
								 'seq': 0.6, 'spo': 0.6}
	ratio_beta_GO = {'cellcycle': 0.3, 'derisi': 0.6, 'eisen': 0.6, 'expr': 0.6, 'gasch1': 0.3, 'gasch2': 0.6,
								'seq': 0.6, 'spo': 0.5}
	ratio_beta_others = {'diatoms': 0.3, 'enron': 0.6, 'imclef07a': 0.6, 'imclef07d': 0.6}

	ratio_beta = {'FUN': ratio_beta_FUN, 'GO': ratio_beta_GO, 'others': ratio_beta_others}

	lrs_FUN = {'cellcycle': 1e-4, 'derisi': 1e-4, 'eisen': 1e-4, 'expr': 1e-4, 'gasch1': 1e-4, 'gasch2': 1e-4,
			   'seq': 1e-4, 'spo': 1e-4}
	lrs_GO = {'cellcycle': 1e-4, 'derisi': 1e-4, 'eisen': 1e-4, 'expr': 1e-4, 'gasch1': 1e-4, 'gasch2': 1e-4,
			  'seq': 1e-4, 'spo': 1e-4}
	lrs_others = {'diatoms': 1e-5, 'enron': 1e-5, 'imclef07a': 1e-5, 'imclef07d': 1e-5}
	lrs = {'FUN': lrs_FUN, 'GO': lrs_GO, 'others': lrs_others}
	epochss_FUN = {'cellcycle': 500, 'derisi': 500, 'eisen': 200, 'expr': 100, 'gasch1': 300, 'gasch2': 300, 'seq': 30,
				   'spo': 300}
	epochss_GO = {'cellcycle': 500, 'derisi': 200, 'eisen': 300, 'expr': 200, 'gasch1': 200, 'gasch2': 200, 'seq': 30,
				  'spo': 300}
	epochss_others = {'diatoms': 800, 'enron': 300, 'imclef07a': 500, 'imclef07d': 300}
	epochss = {'FUN': epochss_FUN, 'GO': epochss_GO, 'others': epochss_others}


	epochss_FUN_g = {'cellcycle': 1000, 'derisi': 200, 'eisen': 200, 'expr': 100, 'gasch1': 300, 'gasch2': 300, 'seq': 300,
				   'spo': 1000}
	epochss_GO_g = {'cellcycle': 1000, 'derisi': 200, 'eisen': 200, 'expr': 200, 'gasch1': 300, 'gasch2': 300, 'seq': 300,
				  'spo': 300}
	epochss_others_g = {'diatoms': 1000, 'enron': 1000, 'imclef07a': 1000, 'imclef07d': 1000}
	epoch_g = {'FUN': epochss_FUN_g, 'GO': epochss_GO_g}

	Nh_FUN = {'cellcycle': 1, 'derisi': 1, 'eisen': 1, 'expr': 1, 'gasch1': 2, 'gasch2': 2,
					   'seq': 1, 'spo': 1}
	Nh_GO = {'cellcycle': 1, 'derisi': 1, 'eisen': 1, 'expr': 1, 'gasch1': 3, 'gasch2': 2,
					  'seq': 1, 'spo': 1}
	Nh_others = {'diatoms': 1, 'enron': 1, 'imclef07a': 3, 'imclef07d': 3}

	Nh = {'FUN': Nh_FUN, 'GO': Nh_GO, 'others': Nh_others}


	# Set the hyperparameters
	batch_size = 4
	num_layers = 3
	dropout = 0.7
	non_lin = 'relu'
	if ontology == 'others':
		hidden_dim = hidden_dims_others[data]
		output_dim = output_dims_others[data]
		lr = lrs_others[data]
		num_epochs = epochss_others[data]
		num_gnn_layer = num_gnn_layers_others[data]
		num_mlp_layer = num_mlp_layers_others[data]
		graph_pooling_type = graph_pooling_type_others[data]
		neighbor_pooling_type = neighbor_pooling_type_others[data]
		num_epoch_g = epochss_others_g[data]
		learning_pace_w = learning_pace_w_others[data]
		threshold_update_freq = threshold_update_freq_others[data]
		ratio_beta = ratio_beta_others[data]
		Nh = Nh_others[data]

	else:
		hidden_dim = hidden_dims[ontology][data]
		output_dim = output_dims[ontology][data]
		lr = lrs[ontology][data]
		num_epochs = epochss[ontology][data]
		num_epoch_g = epoch_g[ontology][data]
		num_gnn_layer = num_gnn_layers[ontology][data]
		num_mlp_layer = num_mlp_layers[ontology][data]
		graph_pooling_type = graph_pooling_type[ontology][data]
		neighbor_pooling_type = neighbor_pooling_type[ontology][data]
		learning_pace_w = learning_pace_w[ontology][data]
		threshold_update_freq = threshold_update_freq[ontology][data]
		ratio_beta = ratio_beta[ontology][data]
		Nh = Nh[ontology][data]

	weight_decay = 1e-5

	hyperparams = {'batch_size': batch_size, 'num_layers': num_layers, 'dropout': dropout, 'non_lin': non_lin,
				   'hidden_dim': hidden_dim, 'lr': lr, 'weight_decay': weight_decay,
				   'num_gnn_layers': num_gnn_layer, 'num_mlp_layers': num_mlp_layer,
				   'graph_pooling_type': graph_pooling_type,
				   'neighbor_pooling_type': neighbor_pooling_type}

	# Set seed
	seed = args.seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

	# Load the datasets
	if ('others' in args.dataset):
		train, test = initialize_other_dataset(dataset_name, datasets)
		train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool), torch.tensor(test.to_eval,
																								   dtype=torch.bool)
	else:
		train, val, test = initialize_dataset(dataset_name, datasets)
		train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool), torch.tensor(
			val.to_eval, dtype=torch.bool), torch.tensor(test.to_eval, dtype=torch.bool)




	R = np.zeros(train.A.shape)
	np.fill_diagonal(R, 1)
	g = nx.DiGraph(train.A)
	for i in range(len(train.A)):
		descendants = list(nx.descendants(g, i))
		if descendants:
			R[i, descendants] = 1

	R = torch.tensor(R)
	# Transpose to get the ancestors for each node
	R = R.transpose(1, 0)
	R = R.unsqueeze(0).to(device)

	# Rescale data and impute missing data
	if ('others' in args.dataset):
		scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
		imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
	else:
		scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))

		imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
		val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(
			device)

	train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(
		device)
	test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(
		device)

	train_g = train.g_list
	test_g = test.g_list

	print(len(train.terms))

	train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]

	if ('others' not in args.dataset):
		# val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
		for (x, y) in zip(val.X, val.Y):
			train_dataset.append((x, y))

		train_g += val.g_list

		train.A = train.A + val.A
	A = train.A
	dummy_1 = np.ones_like(A)
	dummy_0 = np.zeros_like(A)
	A = np.where(A, dummy_1, dummy_0)
	# pickle.dump(A, open('adj_matrix.pkl', 'wb'))
	# print(A)

	A = A.tolist()
	edge_index = [[], []]
	for i in range(len(A)):
		for j in range(len(A[0])):
			if A[i][j] == 1:
				edge_index[0] += [i]
				edge_index[1] += [j]
	edge_index = torch.tensor(edge_index)
	adjacency_matrix, _ = get_adjacency_matrix(edge_index)
	adjacency_matrix = torch.tensor(adjacency_matrix).to(device)

	test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											  batch_size=batch_size,
											  shuffle=False)

	# We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
	if 'GO' in dataset_name:
		num_to_skip = 4
	else:
		num_to_skip = 1


	hierarchy_size = []
	for i in range(len(train.terms[-1].split('.'))):
		hierarchy_size.append(int(sum([1 if len(term.split('.')) == i + 1 else 0 for term in train.terms])))
	# print(hierarchy_size)
	# Create the model
	criterion = nn.BCELoss()

	gnn = GraphCNN(hyperparams["num_gnn_layers"],
				   hyperparams["num_mlp_layers"],
				   hidden_dim,
				   hidden_dim,
				   output_dim + num_to_skip,
				   hyperparams["dropout"],
				   hyperparams["lr"],
				   hyperparams["graph_pooling_type"],
				   hyperparams["neighbor_pooling_type"],
				   hyperparams["batch_size"],
				   device).to(device)
	print('GNN parameter size:', sum(p.numel() for p in gnn.parameters() if p.requires_grad))

	node_idx = torch.tensor(list(range(output_dim + num_to_skip))).long().to(device)
	# print('outside', output_dim + num_to_skip)
	optimizer_g = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=weight_decay)

	a = time.time()
	for epoch in range(num_epoch_g):

		gnn.train()

		optimizer_g.zero_grad()
		feat, g_feat = gnn(node_idx, train_g[:1])
		reconst_loss = nn.MSELoss()(torch.matmul(feat, feat.T), adjacency_matrix)
		# print(torch.matmul(feat.T, feat).size(), adjacency_matrix.size())
		reconst_loss.backward()
		optimizer_g.step()
		# print('Epoch: {0}, Loss: {1}'.format(epoch, reconst_loss))
	b = time.time()
	print('GNN training time: ', b-a)
	model = ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dim + num_to_skip,
								 hyperparams, R, hierarchy_size, device, Nh).to(device)

	print("Model on gpu", next(model.parameters()).is_cuda)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params_size = sum([np.prod(p.size()) for p in model_parameters])
	print('SPUR parameter size: ', params_size)

	# spl = SPLLoss(n_samples=len(train_loader), n_output=18, device=device, model=model, threshold_epoch=threshold_epoch[data])
	spl = SPLLoss_CL(n_samples=len(train_loader), n_output=18, device=device, model=model,
				  threshold_update_freq=threshold_update_freq, ratio_beta=ratio_beta)
	# spl = SPLLoss_hard(n_samples=len(train_loader), n_output=18, device=device, model=model, threshold_epoch=threshold_epoch[data])
	# spl = SPLLoss_Kmeans(n_samples=len(train_loader), n_output=18, device=device, model=model, threshold_epoch=threshold_epoch[data])
	# spl = SPLLoss_meanshift(n_samples=len(train_loader), n_output=18, device=device, model=model, threshold_epoch=threshold_update_freq)
	print('Start Training')



	train_losses = []
	total_time = 0.0
	print('Training size: {0}, Test size: {1}'.format(len(train_loader), len(test_loader)))

	for epoch in range(num_epochs):
		model.train()
		Loss = []
		Losses = []
		Labels = []

		pred_sum, gt_sum = [], []
		cnt = []

		corr0, corr1, total = [], [], []

		node_feats, g_feat = gnn(node_idx, train_g[:1])
		g_feat = g_feat.detach()


		for i, (x, labels) in enumerate(train_loader):
			a = time.time()
			x = x.to(device)
			labels = labels.to(device)

			# Clear gradients w.r.t. parameters
			optimizer.zero_grad()



			output, match_res, masks = model(x.float(), g_feat, adjacency_matrix=adjacency_matrix)

			# MCLoss
			mc_outputs = []
			mc_targets = []
			losses = []

			target = labels[:, train.to_eval]

			constr_output = get_constr_out(output, R)
			train_output = labels * output.double()
			train_output = get_constr_out(train_output, R)

			train_output = (1 - labels) * constr_output.double() + labels * train_output
			targets_0 = []
			for j in range(len(hierarchy_size)):

				input_0 = train_output[:, train.to_eval].float()
				target_0 = target.float()

				mc_outputs.append(input_0)
				mc_targets.append(target_0)

				loss_0 = criterion(input_0, target_0)
				losses.append(loss_0)
				targets_0.append(target_0)

			prob = spl(losses, epoch, targets_0)
			Labels.append(target_0)

			loss = (losses[-1] * prob).mean() - learning_pace_w * prob

			loss.backward()
			optimizer.step()

			b = time.time()
			running_time = (b - a) * 1000

			predicted0 = output.data > 0.5
			dummy = torch.zeros_like(predicted0)
			correct0 = torch.where(labels == 1, labels == predicted0, dummy)

			pred_sum.append(correct0[:, train.to_eval].sum(dim=0).detach().cpu().numpy())
			gt_sum.append(labels[:, train.to_eval].sum(dim=0).detach().cpu().numpy())
			l_i = labels[:, train.to_eval].sum(dim=0).detach().cpu().numpy()
			cnt += [l_i[0]]

			Loss += [loss]
			Losses += [losses]
			corr0 += [correct0.sum()]
			total += [labels.byte().sum()]

		total_time += running_time

		print('Avg running time per instance: {0} sec, Number of training instances: {1}'
			  .format(running_time, len(train_loader)))

		print(sum(corr0) / sum(total))  # , sum(corr1) / sum(total))

		spl.increase_threshold(Labels=Labels, Losses=Losses, epoch=epoch, device=device, output_dims=output_dim)
		# spl.increase_threshold(epoch=epoch)

		# if epoch % 10 == 0:
		if epoch >= 0:
			print('Epoch: {0}, Loss: {1}'.format(epoch, sum(Loss) / len(Loss)))
			train_losses.append(sum(Loss) / len(Loss))

			inference_time = 0.0
			for i, (x, y) in enumerate(test_loader):

				model.eval()
				a = time.time()
				x = x.to(device)
				y = y.to(device)

				constrained_output = model(x.float(), g_feat)
				predicted = constrained_output.data > 0.5

				# Move output and label back to cpu to be processed by sklearn
				predicted = predicted.to('cpu')
				cpu_constrained_output = constrained_output.to('cpu')
				y = y.to('cpu')

				if i == 0:
					predicted_test = predicted
					constr_test = cpu_constrained_output
					y_test = y
				else:
					predicted_test = torch.cat((predicted_test, predicted), dim=0)
					constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
					y_test = torch.cat((y_test, y), dim=0)

				b = time.time()
				inference_time += (b - a) * 1000

			print('Inference time:', inference_time / len(test_loader))

			score = average_precision_score(y_test[:, test.to_eval], constr_test.data[:, test.to_eval], average='micro')
			f = open('results/' + dataset_name + '.csv', 'a')
			f.write(str(seed) + ',' + str(epoch) + ',' + str(score) + '\n')
			f.close()

	# pickle.dump(train_losses, open('results/loss/{0}/ours.pkl'.format(dataset_name, data), 'wb'))
	# pickle.dump(train_losses, open('results/loss/{0}/original.pkl'.format(dataset_name, data), 'wb'))
	# pickle.dump(train_losses, open('results/loss/{0}/hard_only.pkl'.format(dataset_name, data), 'wb'))
	# pickle.dump(train_losses, open('results/loss/{0}/ours_easiest.pkl'.format(dataset_name, data), 'wb'))
	# pickle.dump(train_losses, open('results/loss/{0}/gin_loss_kmeans.pkl'.format(dataset_name, data), 'wb'))


if __name__ == "__main__":
	main()
