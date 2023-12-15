
import torch
import numpy as np
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

def rootsum(x):
	return torch.sqrt(x).sum()


class SPLLoss_hard(nn.BCELoss):
	def __init__(self, *args, n_samples=0, n_output=0, device='cuda', model=None, threshold_epoch=None, **kwargs):
		super(SPLLoss_hard, self).__init__(*args, **kwargs)
		self.threshold = 1.0
		self.growing_factor = 0.2
		self.records = []
		self.model = model
		self.ce = nn.CrossEntropyLoss()
		self.threshold_epoch = threshold_epoch
		# self.v = torch.zeros(n_samples*4, n_output).int().to(device)
		self.gm = None

	def increase_threshold(self, epoch):
		if self.threshold > sum(self.records) / len(self.records):
			self.threshold = sum(self.records) / len(self.records)
		# else:
		#     self.threshold *= self.growing_factor

		self.records = []
		print('threshold:', self.threshold, epoch)

	def forward(self, losses, epoch):
		# print(super_loss)
		super_loss = [loss.detach().cpu().item() for loss in losses]
		v = super_loss[-1] > self.threshold
		self.records.append(sum(super_loss) / len(super_loss))

		if super_loss[-1] < self.threshold:
			v = 1.0
		else:
			v = self.threshold / super_loss[-1]

		self.records.append(super_loss[-1])

		return v



class SPLLoss(nn.BCELoss):
	def __init__(self, *args, n_samples=0, n_output=0, device='cuda', model=None, threshold_epoch=None, **kwargs):
		super(SPLLoss, self).__init__(*args, **kwargs)
		self.threshold = 1.0
		self.growing_factor = 0.2
		self.records = []
		self.model = model
		self.ce = nn.CrossEntropyLoss()
		self.threshold_epoch = threshold_epoch
		# self.v = torch.zeros(n_samples*4, n_output).int().to(device)
		self.gm = None

	def forward(self, losses, epoch, minority=False):

		loss_feat = np.array([[loss.detach().cpu().item() for loss in losses]])
		if self.gm is not None and epoch <= self.threshold_epoch:
			if minority:
				prob = self.gm.predict_proba(loss_feat)[0][self.minority_class]
			else:
				prob = self.gm.predict_proba(loss_feat)[0][self.majority_class]
		# elif self.gm is None:
		# 	prob = 0.5
		else:
			prob = 1.0

		self.records.append(loss_feat)

		return prob

	def increase_threshold(self, epoch):

		self.records = np.concatenate(self.records, axis=0)
		self.gm = GaussianMixture(n_components=2, random_state=0).fit(self.records)
		labels = self.gm.predict(self.records)

		stats = Counter(labels)
		self.majority_class = stats.most_common()[0][0]
		self.minority_class = stats.most_common()[-1][0]
		# print(stats)
		self.records = []
		print(stats)

	# print('threshold:', self.threshold, epoch)

	def spl_loss(self, super_loss):
		# print(super_loss)
		v = super_loss > self.threshold
		self.records.append(torch.mean(super_loss).detach().cpu().item())
		# self.records.append(super_loss)
		return v.int()

	def spl_loss_diff(self, loss0):
		# v = ((loss1 - loss0) < self.threshold) or ((loss0 - loss1) < self.threshold)

		if loss0 < self.threshold:
			v = 1.0
		else:
			v = self.threshold / loss0

		self.records.append(loss0.detach().cpu().item())

		return v

	def spl_acc(self, acc0, acc1):
		# print(super_loss)
		v = torch.tensor(1).to('cuda') if abs(acc0 - acc1) < self.threshold else torch.tensor(0).to('cuda')
		self.records.append(abs(acc0 - acc1))
		# self.records.append(super_loss)
		return v.int()


class SPLLoss_Kmeans(nn.BCELoss):
	def __init__(self, *args, n_samples=0, n_output=0, device='cuda', model=None, threshold_epoch=None, **kwargs):
		super(SPLLoss_Kmeans, self).__init__(*args, **kwargs)
		self.threshold = 1.0
		self.growing_factor = 0.2
		self.records = []
		self.model = model
		self.ce = nn.CrossEntropyLoss()
		self.threshold_epoch = threshold_epoch
		# self.v = torch.zeros(n_samples*4, n_output).int().to(device)
		self.km = None

	def increase_threshold(self, epoch):
		self.records = np.concatenate(self.records, axis=0)
		self.km = KMeans(n_clusters=2, random_state=0).fit(self.records)
		labels = self.km.predict(self.records)

		stats = Counter(labels)
		self.majority_class = stats.most_common()[0][0]
		self.minority_class = stats.most_common()[1][0]
		# print(stats)
		self.records = []
		print(stats)

	def forward(self, losses, epoch):

		loss_feat = np.array([[loss.detach().cpu().item() for loss in losses]])
		if self.km is not None and epoch <= self.threshold_epoch:
			prob = 1.0 if self.km.predict(loss_feat) == self.majority_class else 0.0
		else:
			prob = 1.0

		self.records.append(loss_feat)

		return prob

class SPLLoss_meanshift(nn.BCELoss):
	def __init__(self, *args, n_samples=0, n_output=0, device='cuda', model=None, threshold_epoch=None, **kwargs):
		super(SPLLoss_meanshift, self).__init__(*args, **kwargs)
		self.threshold = 1.0
		self.growing_factor = 0.2
		self.records = []
		self.model = model
		self.ce = nn.CrossEntropyLoss()
		self.threshold_epoch = threshold_epoch
		# self.v = torch.zeros(n_samples*4, n_output).int().to(device)
		self.km = None

	def increase_threshold(self, epoch):
		self.records = np.concatenate(self.records, axis=0)
		bw = estimate_bandwidth(self.records, quantile=0.2, n_samples=10)
		self.km = MeanShift(bandwidth=bw).fit(self.records)
		labels = self.km.predict(self.records)

		stats = Counter(labels)
		self.majority_class = stats.most_common()[0][0]
		self.minority_class = stats.most_common()[1][0]
		# print(stats)
		self.records = []
		print(stats)

	def forward(self, losses, epoch):

		loss_feat = np.array([[loss.detach().cpu().item() for loss in losses]])
		if self.km is not None and epoch <= self.threshold_epoch:
			prob = 1.0 if self.km.predict(loss_feat) == self.majority_class else 0.0
		else:
			prob = 1.0

		self.records.append(loss_feat)

		return prob


class SPLLoss_CL(nn.BCELoss):
	def __init__(self, *args, n_samples=0, n_output=0, device='cuda', model=None, threshold_update_freq=None, ratio_beta=None, **kwargs):
		super(SPLLoss_CL, self).__init__(*args, **kwargs)
		self.threshold = 1.0
		self.growing_factor = 0.2
		self.records = []
		self.model = model
		self.ce = nn.CrossEntropyLoss()
		self.threshold_update_freq = threshold_update_freq
		# self.v = torch.zeros(n_samples*4, n_output).int().to(device)
		self.flag = False
		self.CL_losses = 0.0
		self.ratio_beta = ratio_beta

	def forward(self, losses, epoch, labels, minority=False):

		# self.CL_losses[
		if not self.flag:
			return 1.0
		else:
			prob = 1.0
			for num_loss, loss in enumerate(losses[-1:]):
				batch_label = labels[num_loss]
				avg_loss_at_lvl_i = torch.sum(torch.matmul(batch_label, self.CL_losses))
				prob *= 1 - torch.sigmoid(loss - avg_loss_at_lvl_i)
			# print(prob)
			return prob

	def increase_threshold(self, Labels, Losses, epoch, device, output_dims):

		if not(epoch % self.threshold_update_freq == 0 and epoch >= 10):
			return
		else:
			past_CL_losses = self.CL_losses
			self.CL_losses = {i: [0.0, 1.0] for i in range(output_dims)}
			self.flag = True
			for i, batch_loss in enumerate(Losses):
				# print('batch_loss: ', batch_loss)
				for j in range(len(Labels[i])):
					if j <= len(batch_loss) - 1:
						loss = batch_loss[j]
						batch_label = Labels[i].detach().cpu().tolist()
						for b_idx, b_label in enumerate(batch_label):
							for idx, label in enumerate(b_label):
								if label == 1:
									if self.CL_losses[idx] == 0.0:
										self.CL_losses[idx] = [loss.detach().cpu().item(), 1.0]
									else:
										self.CL_losses[idx][0] += loss.detach().cpu().item()
										self.CL_losses[idx][1] += 1.0

			# print(len(self.CL_losses), output_dims)
			self.CL_losses = torch.tensor([self.CL_losses[i][0] / self.CL_losses[i][1] for i in range(output_dims)]).view(-1, 1)
			# print(past_CL_losses, self.CL_losses)
			self.CL_losses = self.ratio_beta * past_CL_losses + (1-self.ratio_beta) * self.CL_losses.to(device)

	# print('threshold:', self.threshold, epoch)

	def spl_loss(self, super_loss):
		# print(super_loss)
		v = super_loss > self.threshold
		self.records.append(torch.mean(super_loss).detach().cpu().item())
		# self.records.append(super_loss)
		return v.int()

	def spl_loss_diff(self, loss0):
		# v = ((loss1 - loss0) < self.threshold) or ((loss0 - loss1) < self.threshold)

		if loss0 < self.threshold:
			v = 1.0
		else:
			v = self.threshold / loss0

		self.records.append(loss0.detach().cpu().item())

		return v

	def spl_acc(self, acc0, acc1):
		# print(super_loss)
		v = torch.tensor(1).to('cuda') if abs(acc0 - acc1) < self.threshold else torch.tensor(0).to('cuda')
		self.records.append(abs(acc0 - acc1))
		# self.records.append(super_loss)
		return v.int()

