import numpy as np
from scipy import sparse
# create our example feature matrix
example = np.array(
 [
 [0, 0, 1],
 [1, 0, 0],
 [1, 0, 1]
 ]
)
# convert numpy array to sparse CSR matrix
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix
print(example.data.nbytes)
print(sparse_example.data.nbytes)

# number of rows
n_rows = 100
# number of columns
n_cols = 1000
# create random binary matrix with only 5% values as 1s
example = np.random.binomial(1, p=0.05, size=(n_rows, n_cols))
# print size in bytes
print(f"Size of dense array: {example.nbytes}")
# convert numpy array to sparse CSR matrix
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix
print(f"Size of sparse array: {sparse_example.data.nbytes}")
full_size = (
 sparse_example.data.nbytes +
 sparse_example.indptr.nbytes +
 sparse_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Full size of sparse array: {full_size}")


# one-hot encoding
import numpy as np
from scipy import sparse
# create our example feature matrix
example = np.array(
 [
 [0, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0]
 ]
)

print(f"Size of dense array: {example.nbytes}")
# convert numpy array to sparse CSR matrix
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix

# print size of this sparse matrix
print(f"Size of sparse array: {sparse_example.data.nbytes}")
full_size = (
 sparse_example.data.nbytes +
 sparse_example.indptr.nbytes +
 sparse_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Full size of sparse array: {full_size}")

