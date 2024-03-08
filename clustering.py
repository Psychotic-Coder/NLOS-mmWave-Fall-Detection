import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import csv
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

with open('file_path') as csvfile:
    data_p1 = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))

data_p1[:,1] -= np.amin(data_p1[:,1]) 
data = data_p1

max_time = np.amax(data[:,0])
n_frames = int(np.amax(data[:,1]))

print(n_frames)

x_max, x_min = max(data[:,3]), min(data[:,3])
y_max, y_min = max(data[:,4]), min(data[:,4])
z_max, z_min = max(data[:,5]), min(data[:,5])

fig2D = plt.figure(figsize = (10,6))
xy2D = fig2D.add_subplot(1,2,1)
xz2D = fig2D.add_subplot(1,2,2)

#TODO: How to find human???
#Clustering ki MKB
def update2D(i):
    data_s = data[data[:, 1] == i]
    model = DBSCAN(eps=0.04, min_samples=4)
    cluster = model.fit(data_s[:,3:6])
    pred = cluster.labels_
    frame_intensity = data_s[:, -1]

    weight = ((frame_intensity - np.amin(frame_intensity)) / (np.amax(frame_intensity) - np.amin(frame_intensity)))

    xy2D.clear()
    xz2D.clear()

    xy2D.set_xlim(x_min, x_max)
    xy2D.set_ylim(y_min, y_max)
    xz2D.set_xlim(x_min, x_max)
    xz2D.set_ylim(z_min, z_max)

    xy2D.scatter(data_s[:,3], data_s[:,4], c=(weight*pred), s=10)
    xz2D.scatter(data_s[:,3], data_s[:,5], c=(weight*pred), s=10)

ani2D = anim.FuncAnimation(fig2D, update2D, frames=n_frames, interval=20)
plt.show()

fig3D = plt.figure(figsize = (10,6))
xyz3D_noise = fig3D.add_subplot(1,2,1, projection='3d')
xyz3D_filtered = fig3D.add_subplot(1,2,2, projection='3d')

#TODO: Check on removing noise
def update3D(i):
    data_s = data[data[:, 1] == i]
    model = DBSCAN(eps=0.04, min_samples=4)
    cluster = model.fit(data_s[:,3:6])
    pred = cluster.labels_
    frame_intensity = data_s[:, -1]

    weight = ((frame_intensity - np.amin(frame_intensity)) / (np.amax(frame_intensity) - np.amin(frame_intensity)))

    # print(len(set(pred)), set(pred))
    data_s = np.hstack((data_s, (weight*pred).reshape(-1, 1)))
    data_filt = data_s[data_s[:, -1] > 0]
    # print(data_s.shape, (weight*pred).reshape(-1, 1).shape)
    xyz3D_noise.clear()
    xyz3D_filtered.clear()

    xyz3D_noise.set_xlim3d(x_min, x_max)
    xyz3D_noise.set_ylim3d(y_min, y_max)
    xyz3D_noise.set_zlim3d(z_min, z_max)
    xyz3D_filtered.set_xlim3d(x_min, x_max)
    xyz3D_filtered.set_ylim3d(y_min, y_max)
    xyz3D_filtered.set_zlim3d(z_min, z_max)

    sc_noise = xyz3D_noise.scatter(data_s[:,3], data_s[:,4], data_s[:,5], s=10)
    sc_filt = xyz3D_filtered.scatter(data_filt[:,3], data_filt[:,4], data_filt[:,5], s=10)

    sc_noise.set_array(data_s[:, -1])
    sc_filt.set_array(data_filt[:, -1])

    # xyz3D_noise.view_init(azim=0)
    # xyz3D_filtered.view_init(azim=0)

ani3D = anim.FuncAnimation(fig3D, update3D, frames=n_frames, interval=20)
plt.show()