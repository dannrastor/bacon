
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

plt.rcParams['figure.figsize'] = (10, 4)


def load_frame(path, th):
    fname = '{}.pmf_THL1.pmf'.format(str(th).zfill(3))
    full_path = os.path.join(path, fname)
    print('Loading ', full_path)
    return select_roi(np.loadtxt(full_path), fov)

def get_ff_frame(th):
    frame = load_frame(config['frame_path'], th)
    ff = load_frame(config['ff_path'], th)
    return np.divide(frame, ff, out=np.zeros_like(frame), where=ff!=0)

def load_rois():
    rois = config['rois']
    result = {}
    print('Loading ROIs, {} found:'.format(len(rois)))
    for roi in rois:
        name = roi['name']
        result[name] = (roi['xmin'], roi['xmax'], roi['ymin'], roi['ymax'])
        print('{}: {}<x<{}; {}<y<{}'.format(name, *result[name]))
    print('')
    return result

def select_roi(frameset, coords):
    x1, x2, y1, y2 = coords
    return frameset[y1:y2, x1:x2]
    

with open('view_config.yaml') as f:
    config = yaml.safe_load(f)

a, b, t = config['a'], config['b'], config['t']
p1_min, p1_max = config['p1_min'], config['p1_max']

fovs = config['fov']
fov = fovs['xmin'], fovs['xmax'], fovs['ymin'], fovs['ymax']
rois = load_rois()


plt.figure('raw uncorrected')
plt.imshow(load_frame(config['frame_path'], 40), origin='lower', cmap='hot')
plt.colorbar()
plt.title('Counts at raw image at TH = 40')



frame_a = get_ff_frame(a)
frame_b = get_ff_frame(b)
frame_t = get_ff_frame(t)

#distr = (frame_a - frame_b) / (frame_t - frame_b)
nn = frame_a - frame_b
dd = frame_t - frame_b
distr = np.divide(nn, dd, out=np.zeros_like(nn), where=dd!=0)

print ()
for roi in rois:
    x1, x2, y1, y2 = rois[roi]
    xx1, xx2, yy1, yy2 = fov

    r = distr[y1-yy1:y2-yy1, x1-xx1:x2-xx1]
    print(roi)
    r_sz = r.size
    r_uf = np.count_nonzero(r < p1_min)
    r_of = np.count_nonzero(r > p1_max)
    r_r = r_sz - r_uf - r_of
    
    print(r.size, ' pixels total')
    print(r_uf, ' pixels with p1 below ', p1_min)
    print(r_r, ' pixels within given range')
    print(r_of, ' pixels with p1 above ', p1_max)
    print()
    


distr[distr < p1_min] = p1_min
distr[distr > p1_max] = p1_min
#distr[distr != 0] = 1

plt.figure('distr')
plt.imshow(distr, origin='lower', cmap='hot')

name = 'Distribution (a = {}, b = {}, t = {}, {} < p1 < {})'.format(a, b, t, p1_min, p1_max)
plt.title(name)
plt.colorbar()


plt.show()



 

 
