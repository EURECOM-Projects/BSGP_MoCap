import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns

def get_lower_triangular_from_diag(D):
    lengthscales = np.full((D,), D**0.5, dtype=np.float64) # D
    Lambda = np.diag(1/(lengthscales**2)) # D x D
    L = scipy.linalg.cholesky(Lambda, lower=True) # D x D
    L_array = tfp.math.fill_triangular_inverse(L, upper=False) # D*(D+1)/2
    return L_array  #tf.tile(L_array[tf.newaxis, :], [T, 1]) # T x D*(D+1)/2

def get_lower_triangular_uniform_random(d):
    L = tf.cast(tf.fill([d*(d+1)//2,], d**0.5), dtype=tf.float64)#np.sqrt(d)*np.ones((d*(d+1)//2,1))
    return L

def mocap_sensors_info(features):
    sensors_names = ['root','lowerback','upperback','thorax','lowerneck','upperneck','head','rclavicle','rhumerus','rradius','rwrist','rhand','rfingers','rthumb','lclavicle','lhumerus','lradius','lwrist','lhand','lfingers','lthumb','rfemur','rtibia','rfoot','rtoes','lfemur','ltibia','lfoot','ltoes']
    sensors_measures = [6,3,3,3,3,3,3,2,3,1,1,2,1,2,2,3,1,1,2,1,2,3,1,2,1,3,1,2,1]
    s_names_measures = []
    for i, sensor_name in enumerate(sensors_names):
        s_names_measures.append([sensor_name] * sensors_measures[i])
    mask = [31, 32, 33, 34, 35, 43, 44, 45, 46, 47, 54, 61]
    mask_sensors_names = [11,12,13,18,19,20,24,28]
    s_names_measures = np.array([item for s_names_measures in s_names_measures for item in s_names_measures])
    s_names_measures_50 = np.delete(s_names_measures, mask, 0).tolist()
    s_sep_50 = np.cumsum(np.delete(np.array(sensors_measures), mask_sensors_names, 0))
    sensors_measures_21 = np.insert(s_sep_50, 0, 0)
    sensors_names_21 = list(dict.fromkeys(s_names_measures_50))
    measures_per_sensor_50 = np.delete(np.array(sensors_measures), mask_sensors_names, 0)
    if features == 'full':
        return s_names_measures_50, measures_per_sensor_50
    else:
        return sensors_names_21, measures_per_sensor_50

def plot_kernel_matrix(Kxx, filename, width=7, height=7):
    fig, ax = plt.subplots(1, 1, figsize=(width, width))
    sns.heatmap(Kxx, annot=False, square=True, cmap='vlag', vmax=np.max(Kxx), vmin=-np.max(Kxx), center=0, ax=ax)
    plt.savefig('..results/%s.png'%filename, dpi=300)

def plot_precision_matrix(path='./results', filename='precision_matrix', P=None, features='full'):
    sensors_names, _ = mocap_sensors_info(features)
    image_size = 7 if features == 'small' else 15 
    fig, ax = plt.subplots(1, 1, figsize=(image_size, image_size))
    sns.heatmap(P, annot=False, square=True, cmap='vlag', vmax=np.max(P), vmin=-np.max(P), center=0, ax=ax)
    ax.set_xticklabels(sensors_names)
    ax.set_yticklabels(sensors_names)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=360)
    plt.savefig('%s/%s.png'%(path,filename), dpi=300)