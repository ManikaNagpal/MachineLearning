# MachineLearning

Created on Tue Sep 18 21:13:10 2018


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples = 100, random_state=0, cluster_std=0.4)

plt.scatter(x[:,0], x[:,1], s = 50)
plt.show()

x,y = make_blobs(n_samples = 1000, random_state=0, cluster_std=3)

kmeans = KMeans(3)
kmeans.fit(x)

y_means = kmeans.predict(x)

# plt.scatter(x[:,0], x[:,1], s = 50)

#plt.scatter(x[:,0], x[:,1], c=y_means, s = 50, cmap = 'rainbow')

#plt.show()

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["r.","g.","c."]

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]] )
    
plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='b', s = 120, zorder=10)
plt.show()

centroids
