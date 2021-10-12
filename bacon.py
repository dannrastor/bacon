
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

plt.rcParams['figure.figsize'] = (12, 8)


def load_frameset(path):
    files = filter(lambda s : s.endswith('THL1.pmf'), os.listdir(path))
    files = list(sorted(files))
    
    thr_min = int(files[0][:3])
    thr_max = int(files[-1][:3])
    thr_step = int((thr_max - thr_min) / (len(files) - 1))
    
    print('Reading {}'.format(path))
    print('{} files detected'.format(len(files)))
    print('THR goes from {} to {} with a step of {}'.format(thr_min, thr_max, thr_step))

    frames = []
    for f in files:
        full_path = os.path.join(path, f)
        print('Loading ', full_path, end='\r')
        frames.append(np.loadtxt(full_path))
    
    print('\nDone!!!\n')
    result = np.swapaxes(np.stack(frames), 1, 2)
    
    return result, (thr_min, thr_max + 1, thr_step)


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
    return frameset[:, x1:x2, y1:y2]
    
def get_index(thr):
    return np.where(th_array == thr)[0][0]

with open('config.yaml') as f:
    config = yaml.safe_load(f)



a, b = config['a'], config['b']
reference_roi = config['reference_roi']
print('a={} b={}'.format(a, b))
rois = load_rois()
frameset, thrs = load_frameset(config['frame_path'])
ff_set, _ = load_frameset(config['ff_path'])


th_array = np.arange(*thrs)
t = np.arange(a, b, thrs[2])



r = select_roi(frameset, rois[reference_roi])
rff = select_roi(ff_set, rois[reference_roi])
rd = np.divide(r, rff, out=np.zeros_like(r), where=rff!=0)
rda = np.average(rd, axis=(1, 2))
st = rda[get_index(a):get_index(b)]
sa = rda[get_index(a)]
sb = rda[get_index(b)]

reference_p = (sa - sb) / (st - sb)



for roi in rois:

    r = select_roi(frameset, rois[roi])
    rff = select_roi(ff_set, rois[roi])
    
    rd = np.divide(r, rff, out=np.zeros_like(r), where=rff!=0)
    rda = np.average(rd, axis=(1, 2))
    
    ra = np.average(r, axis=(1, 2))
    rffa = np.average(rff, axis=(1, 2))



    plt.figure('Threshold scans')
    plt.title('ROI-averaged threshold scans')
    plt.xlabel('TH')
    plt.ylabel('Counts')
    plt.plot(th_array, ra, label=roi)
    plt.plot(th_array, rffa, label=roi+'_ff')
    plt.grid()
    plt.legend()


    plt.figure('FF-corrected threshold scans')
    plt.title('ROI-averaged FF-corrected scans')
    plt.xlabel('TH')
    plt.ylabel('Counts')
    plt.plot(th_array, rda, label=roi)
    plt.grid()
    plt.legend()


    st = rda[get_index(a):get_index(b)]
    sa = rda[get_index(a)]
    sb = rda[get_index(b)]

    p1 = (sa - sb) / (st - sb)
    
    if roi == reference_roi:
        reference_p = p1

    plt.figure('Parameters')

    plt.subplot(1,2,1)
    plt.title('P1 = (a - b) / (t - b)')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.grid()
    plt.plot(t[:-10], p1[:-10], label=roi)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title('Difference with reference ROI ({})'.format(reference_roi))
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.plot(t[:-10], p1[:-10] - reference_p[:-10], label=roi)
    plt.grid()
    plt.legend()

plt.show()




plt.grid()
plt.legend()
plt.show()
 

