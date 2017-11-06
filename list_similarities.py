import os
import numpy as np

from sklearn.cluster import MiniBatchKMeans, SpectralClustering, DBSCAN
from pyemd import emd

names = []
data = []
states = []
states_norm = []

np.set_printoptions(threshold='nan', precision=3, suppress=True)

for game in [x[0] for x in os.walk('chk')][1:]:
  if 'zelda' not in game:
    names.append(game)
    #dat = [ 1.0 ] #np.load(game + '/__EWCstrength.npy')
    #dat = dat / np.sum(np.abs(dat))
    #data.append(dat)
names = sorted(names)
    
for game in names:
    print game
    dat = np.load(game + '/states.npy')[-10000:]
    #print dat[0]
    dat = np.mean(dat, axis=0)
    print np.sum(np.abs(dat))
    states.append(dat)
    dat = dat / np.sum(np.abs(dat))
    states_norm.append(dat)

print names

dats = []
for game in names:
    dat = np.load(game + '/states.npy')[-10000:]
    #dat = dat[np.random.choice(dat.shape[0], 10000, replace=False)]
    dats.append(dat)

def hist(array, items=100):
    counts = np.zeros((items))
    for a in array:
        counts[a] += 1
    return counts
    
num_bins = 8
alldata = np.concatenate(dats)

kmeansclass = MiniBatchKMeans(num_bins)
kmeansclass.fit(alldata)
labels = []
for dat in dats:
    label = kmeansclass.predict(dat)
    counts = hist(label, num_bins)
    print counts
    labels.append(counts)


#specclass = DBSCAN(num_bins)
#l = specclass.fit_predict(alldata)
#l = l + 2
#num_bins = np.max(l) + 2
#labels = []
#for dat in dats:
#    i = len(dat)
#    label = l[:i] ; l = l[i:]
#    counts = hist(label, num_bins)
#    print counts
#    labels.append(counts)

dist = np.ones((num_bins, num_bins)) - np.eye(num_bins)
for i, a in enumerate(kmeansclass.cluster_centers_):
    for j, b in enumerate(kmeansclass.cluster_centers_):
        dist[i,j] = np.sqrt(np.sum((a - b) ** 2))

for i, name_a in enumerate(names):
    l = '{:40}'.format(name_a)
    for j, name_b in enumerate(names):
      if i >= j:
        #dist = np.ones((num_bins, num_bins)) - np.eye(num_bins)
        sim = emd(labels[i], labels[j], dist)/10000
        l = l + '{:6.5f}, '.format(sim)
    print(l)

print(' ')

#for i, name_a in enumerate(names):
#    l = '{:40}'.format(name_a)
#    for j, name_b in enumerate(names):
#      if i >= j:
#        dat_a = data[i] ; dat_b = data[j]
#        sim = np.sum((dat_a - dat_b)**2)
#        l = l + '{:6.5f}, '.format(sim)
#        
#    print(l)

print(' ')

for i, name_a in enumerate(names):
    l = '{:40}'.format(name_a)
    for j, name_b in enumerate(names):
      if i >= j:
        dat_a = states[i] ; dat_b = states[j]
        sim = np.sum((dat_a - dat_b)**2)
        l = l + '{:6.5f}, '.format(sim)
    print(l)
    
print(' ')

for i, name_a in enumerate(names):
    l = '{:40}'.format(name_a)
    for j, name_b in enumerate(names):
      if i >= j:
        dat_a = states_norm[i] ; dat_b = states_norm[j]
        sim = np.sum((dat_a - dat_b)**2)
        l = l + '{:6.5f}, '.format(sim)
    print(l)
