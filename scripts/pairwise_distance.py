from dask_ml.metrics import pairwise_distances, euclidean_distances
from dask.distributed import Client, progress
from dask_jobqueue import SGECluster
from dask import array as da
import h5py
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

cluster = SGECluster(cores=8, memory="16 GB", queue="lThC.q")
cluster.scale(4)
print(cluster.job_script())

client = Client(cluster)

h5_data = h5py.File(
    "/data/sao/klee/projects/rotconml/data/processed/newset-processed-split-data.hd5",
    mode="r",
)["full"]

coulomb_matrix = da.from_array(h5_data["coulomb_matrix"])
eigenvalues = da.from_array(h5_data["eigenvalues"])

n_mols = eigenvalues.shape[0]
n_atoms = coulomb_matrix.shape[1]

reshaped_coulomb = coulomb_matrix.reshape(n_mols, n_atoms ** 2)

# Reshape for histogram binning
coulomb_distances = euclidean_distances(reshaped_coulomb).reshape(1, -1)
eigen_distances = euclidean_distances(eigenvalues).reshape(1, -1)

# Histogram the distances
coulomb_hist, coulomb_bins = da.histogram(
    coulomb_distances, bins=np.linspace(0.0, 500, 50)
)
eigen_hist, eigen_bins = da.histogram(eigen_distances, bins=np.linspace(0.0, 500, 50))

# Normalize the histograms, and chop off the last bin
eigen_bins = eigen_bins[:-1]
coulomb_bins = coulomb_bins[:-1]

print("Computing arrays")

coulomb_hist = coulomb_hist / da.sum(coulomb_hist).compute()
progress(coulomb_hist)
eigen_hist = eigen_hist / da.sum(eigen_hist).compute()
progress(eigen_hist)

print("Done")

# Save the arrays to disk
h5_pairs = h5py.File(
    "/data/sao/klee/projects/rotconml/data/processed/coulomb_pairwise.hd5", mode="w"
)
coulomb_group = h5_pairs.create_group("coulomb_matrix")
coulomb_group.create_dataset("histogram", data=coulomb_hist)
coulomb_group.create_dataset("bins", data=coulomb_bins)

eigen_group = h5_pairs.create_group("eigen_matrix")
eigen_group.create_dataset("histogram", data=eigen_hist)
eigen_group.create_dataset("bins", data=eigen_bins)
