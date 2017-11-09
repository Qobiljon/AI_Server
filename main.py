import os
import pickle
import numpy
from pybrain3.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain3.structure import LinearLayer
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork


class Tools:
	@staticmethod
	def objdump(dmp_obj, dmp_filename):
		dump = open(dmp_filename, 'wb')
		pickle.dump(dmp_obj, dump)
		dump.close()

	@staticmethod
	def objrecv(dmp_filename):
		dump = open(dmp_filename, 'rb')
		dmp_obj = pickle.load(dump)
		dump.close()
		return dmp_obj


class CategoryAdvisor:
	# region Notes on using this class
	# In order to use dataset, history must contain >2*OBSERVE_LENGTH (which is >20)
	# endregion

	# region Constants
	PROJECT_DIR = 'myweek-ai-data'

	NET_EXT = 'netdmp'
	DST_EXT = 'dstdmp'

	OBSERVE_LENGTH = 5
	PREDICT_LENGTH = 1
	# endregion

	# region Variables
	bprnetw = None
	dataset = None
	user = None
	category_id = None

	# endregion

	@staticmethod
	def create(user, category_id, network=None, dataset=None):
		res = CategoryAdvisor()

		res.user = user
		res.category_id = category_id

		if network is None:
			res.bprnetw = buildNetwork(CategoryAdvisor.OBSERVE_LENGTH, 20, CategoryAdvisor.PREDICT_LENGTH, outclass=LinearLayer, bias=True, recurrent=True)
		else:
			res.bprnetw = network

		if dataset is None:
			res.dataset = SupervisedDataSet(CategoryAdvisor.OBSERVE_LENGTH, CategoryAdvisor.PREDICT_LENGTH)
		else:
			res.dataset = dataset

		return res

	@staticmethod
	def recover(user, category_id):
		if not CategoryAdvisor.is_backed_up(user, category_id):
			return False

		res = CategoryAdvisor()

		res.user = user
		res.category_id = category_id
		res.bprnetw = Tools.objrecv('~/%s/%s.%s' % (res.PROJECT_DIR, user.username, res.NET_EXT))
		res.dataset = Tools.objrecv('~/%s/%s.%s' % (res.PROJECT_DIR, user.username, res.DST_EXT))

		return res

	@staticmethod
	def is_backed_up(user, category_id):
		return \
			os.path.exists('~/%s' % CategoryAdvisor.PROJECT_DIR) and \
			os.path.isfile('~/%s/%s-%d.%s' % (CategoryAdvisor.PROJECT_DIR, user.username, category_id, CategoryAdvisor.NET_EXT)) and \
			os.path.isfile('~/%s/%s-%d.%s' % (CategoryAdvisor.PROJECT_DIR, user.username, category_id, CategoryAdvisor.DST_EXT))

	def backup(self):
		# create backup files in home directory
		if not os.path.exists('~/%s' % self.PROJECT_DIR):
			os.makedirs('~/%s' % self.PROJECT_DIR)

		Tools.objdump(dmp_obj=self.bprnetw, dmp_filename='~/%s/%s-%d.%s' % (self.PROJECT_DIR, self.user, self.category_id, self.NET_EXT))
		Tools.objdump(dmp_obj=self.dataset, dmp_filename='~/%s/%s-%d.%s' % (self.PROJECT_DIR, self.user, self.category_id, self.DST_EXT))

	def retrain_complete(self, history_complete):
		if len(history_complete) >= CategoryAdvisor.OBSERVE_LENGTH:
			self.dataset.clear()

			data_length = len(history_complete)

			for n in range(data_length):
				if n + (CategoryAdvisor.OBSERVE_LENGTH - 1) + CategoryAdvisor.PREDICT_LENGTH < data_length:
					self.dataset.addSample(history_complete[n:n + CategoryAdvisor.OBSERVE_LENGTH], history_complete[n + 1:n + 1 + CategoryAdvisor.PREDICT_LENGTH])

			trainer = BackpropTrainer(self.bprnetw, self.dataset)
			trainer.trainEpochs(100)

			return True
		else:
			return False

	def retrain_single(self, value):
		inp = numpy.append(self.dataset['input'][-1][1:], self.dataset['target'][-1])
		self.dataset.addSample(inp, [value])

		trainer = BackpropTrainer(self.bprnetw, self.dataset)
		trainer.trainEpochs(100)

	def calculate(self):
		ts = UnsupervisedDataSet(CategoryAdvisor.OBSERVE_LENGTH, )
		ts.addSample(self.dataset['input'][-1])
		return int(self.bprnetw.activateOnDataset(ts)[0][0])


if __name__ == '__main__':
	advisor = CategoryAdvisor.create(None, None)
	advisor.retrain_complete([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
	print(advisor.calculate())
