import numpy as np
from sklearn import preprocessing

# create random 1-d array with 1001 different categories (int)
example = np.random.randint(1000, size=1000000)

# initialize oneHotEncoder from scikit-learn
# keep sparse = False to get dense array
ohe = preprocessing.OneHotEncoder(sparse=False)

# fit and transform data with dense one hot encoder

ohe_example = ohe.fit_transform(example.reshape(-1, 1))

# print size in bytes for dense array
print(f"Size of dense array: {ohe_example.nbytes}")

# initialize OneHoteEncoder from scikit learn
# keep sparse = True
ohe = preprocessing.OneHotEncoder(sparse=True)

# fit and transform with sparse one-hot encoder
ohe_example = ohe.fit_transform(example.reshape(-1, 1))

# print size of sparse matrix
print(f"Size of sparse array: {ohe_example.data.nbytes}")

full_size = {
    ohe_example.data.nbytes +
    ohe_example.indptr.nbytes + ohe_example.indices.nbytes
}

# print full size of sparse matrix
print(f"Full size of sparse array: {full_size}")