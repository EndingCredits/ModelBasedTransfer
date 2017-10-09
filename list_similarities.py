import os
import numpy as np

names = []
data = []
states = []
states_norm = []

np.set_printoptions(threshold='nan', precision=3, suppress=True)

for game in [x[0] for x in os.walk('chk')][1:]:
  #if 'objects' not in game:
    names.append(game)
    dat = np.load(game + '/__EWCstrength.npy')
    dat = dat / np.sum(np.abs(dat))
    data.append(dat)
    
for game in names:
    dat = np.load(game + '/states.npy')
    dat = np.mean(dat, axis=0)
    print dat#np.mean(np.abs(dat))
    states.append(dat)
    dat = dat / np.sum(np.abs(dat))
    states_norm.append(dat)

print names
  
for i, name_a in enumerate(names):
    l = '{:40}'.format(name_a)
    for j, name_b in enumerate(names):
      if i >= j:
        dat_a = data[i] ; dat_b = data[j]
        sim = np.sum((dat_a - dat_b)**2)
        l = l + '{:6.5f}, '.format(sim)
        
    print(l)

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
