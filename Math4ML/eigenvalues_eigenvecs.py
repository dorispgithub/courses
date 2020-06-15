# Eigenvalues
M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
vals, vecs = np.linalg.eig(M)
vals



# Eigenvectors - Note, the eigenvectors are the columns of the output.
M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
vals, vecs = np.linalg.eig(M)
vecs
