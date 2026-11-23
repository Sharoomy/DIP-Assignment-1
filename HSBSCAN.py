#imports
import hdbscan

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#clean the data

#train hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=True, leaf_size=40, memory=Memory(None),
    metric='euclidean', min_cluster_size=5, min_samples=None, p=None)


#evaluate hdbscan

labels = clusterer.labels_
silhouette_avg = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, labels)

#evaluation metrics
print("Silhouette Coefficient: {:.2f}".format(silhouette_avg))
print("Adjusted Rand Index: {:.2f}".format(ari))

#visualise clusters

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_,
            cmap='viridis', s=50, alpha=0.7, edgecolors='k')
plt.colorbar()
plt.title('HDBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()