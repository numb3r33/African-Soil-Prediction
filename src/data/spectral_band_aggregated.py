import pandas as pd
import re

from collections import defaultdict

spectral_band = re.compile(r'([a-z]+)([0-9]+)')

class Data:

	def __init__(self, train, test):
		self.train = train
		self.test = test

	def prepare(self):
		train_new = {}
		test_new = {}
		
		band_dict = self.group_by_wavelength()

		for k, v in band_dict.items():
			train_new[k] = self.train[v].mean(axis=1)
			test_new[k] = self.test[v].mean(axis=1)

		train_ = pd.DataFrame(train_new)
		test_ = pd.DataFrame(test_new)

		return train_, test_

	def group_by_wavelength(self):
		"""
		Group features based on the band

		e.g m7423.12, m7542.24 etc map to 7000
		"""

		spectral_features = self.train.columns[1:-21]
		
		band_dict = defaultdict(list)

		for col in spectral_features:
			match = spectral_band.match(col)

			alpha, numeric = match.groups()
			n = len(numeric)

			band_dict[int(numeric[0]) * (10 ** (n - 1))].append(col)
		
		return band_dict