#!/usr/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert, load
import time
import torch
print('GPU available: %s' % torch.cuda.is_available())

# Create some random data for binary classification
num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)

# Create and train a model (scikit-learn RandomForestClassifier in this case)
skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
skl_model.fit(X, y)

# Use Hummingbird to convert the model to PyTorch
model = convert(skl_model, 'pytorch')

# Run predictions on CPU
start_time = time.time()
skl_model.predict(X)
finish_time = time.time()
print(finish_time - start_time)
# Run predictions on GPU
start_time = time.time()
model.to('cuda')
model.predict(X)
finish_time = time.time()
print(finish_time - start_time)
