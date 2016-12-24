"""Analyze the radiometric measurements I did in Elbit.

This is basically used for comparing between cameras.
The script accomulates the measurments of RGB values per exposure.
It calculates a liner fit between the two (per channel) and stores
the linear coefficient (ratio), i.e:
    R = coef[0] * exposure
    G = coef[1] * exposure
    B = coef[2] * exposure
"""
from __future__ import division
from CameraNetwork.integral_sphere import measureRGB
from CameraNetwork.integral_sphere import parseData
import cPickle
import glob
from itertools import chain
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.io as sio
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

CAM_MAP = {
    "4102820388": "cam_101",
    "4102820378": "cam_102",
    "4102820395": "cam_103",
    "4102820374": "cam_104",
    "4102820391": "cam_105",
    "4102820375": "cam_106",
    "4102820392": "cam_107",
}


def visualize():
    """Visualize the measurments of cameras per distance."""
    
    #
    # Load the measurements
    #
    plt.figure()    
    for cam_id in (
        '4102820374',
        '4102820375',
        '4102820378',
        '4102820388',
        '4102820391'):
        base_path =  r'elbit_calibration\{}'.format(cam_id)
        
        df = dict(
            Distance=[],
            Intensity=[],
            Candela=[],
            Gain=[],
            Exposure=[],
            R=[],
            G=[],
            B=[]
        )
        for path in glob.glob(os.path.join(base_path, '*')):
            if not os.path.isdir(path):
                continue
            print(path)
            distance, intensity, candela = parseData(path)
            for measure_file in glob.glob(os.path.join(path, '*.mat')):
                d = sio.loadmat(measure_file)
                exposure = d['exposure'][0, 0]
                gain = measure_file.split('.')[0].split('_')[-1] == 'True'
                r, g, b = measureRGB(d['array'])
                
                df['Distance'].append(distance)
                df['Intensity'].append(intensity)
                df['Candela'].append(candela)
                df['Gain'].append(gain)
                df['Exposure'].append(exposure)
                df['R'].append(r)
                df['G'].append(g)
                df['B'].append(b)
                
                model = make_pipeline(
                    PolynomialFeatures(1),
                    linear_model.RANSACRegressor(random_state=0, residual_threshold=5)
                )
    
        df = pd.DataFrame.from_dict(df)
        df[(df['Gain']==True)&(df['Distance']==5.7)][['Exposure', 'R']]
        
        for d in np.unique(df['Distance']):
            plt.plot(
                df[(df['Gain']==False)&(df['Distance']==d)]['Exposure'],
                df[(df['Gain']==False)&(df['Distance']==d)]['B'],
            )
        
    plt.show()


def process_cam(base_path):
    """Collect all exposures of a camera."""
    
    mat_files = chain.from_iterable(glob.glob(os.path.join(dirpath, '*.mat')) for dirpath, _, _ in os.walk(base_path))
    
    df = dict(
        Gain=[],
        Exposure=[],
        R=[],
        G=[],
        B=[]
    )    

    for mat_file in mat_files:
        d = sio.loadmat(mat_file)
        exposure = d['exposure'][0, 0]
        gain = mat_file.split('.')[0].split('_')[-1] == 'True'
        r, g, b = measureRGB(d['array'])
        

        df['Gain'].append(gain)
        df['Exposure'].append(exposure)
        df['R'].append(r)
        df['G'].append(g)
        df['B'].append(b)
        
    df = pd.DataFrame.from_dict(df)
    
    return df


def main():
    """Process all cameras."""

    base_paths = list(glob.glob(r'elbit_calibration\*'))
    cam_ids = [base_path.split('\\')[-1] for base_path in base_paths]

    #
    # Accomulate all exposures per camera.
    #
    dfs = Parallel(n_jobs=-1, verbose=100)(delayed(process_cam)(base_path) for base_path in base_paths)

    coefs = dict()
    for cam_id, df in zip(cam_ids, dfs):
        exposure = df[(df['Gain']==False)]['Exposure']
        
        tmp = []
        RGB = df[(df['Gain']==False)][['R', 'G', 'B']].values
        for C, color in zip(RGB.T, ('R', 'G', 'B')):
            model = make_pipeline(
                PolynomialFeatures(1),
                linear_model.RANSACRegressor(random_state=0, residual_threshold=5)
            )
            
            model.fit(exposure.values.reshape(-1, 1), C)
            tmp.append(model.steps[1][1].estimator_.coef_[1])
            plt.figure()
            plt.plot(model.steps[1][1].estimator_.coef_[1] * exposure.values, 'r')
            plt.plot(C, 'b')
            plt.title("{}, {}, {}".format(cam_id, CAM_MAP.get(cam_id), color))
            plt.show()
        coefs[cam_id] = tmp
    
    with open('elbit_results.pkl', 'wb') as f:
        cPickle.dump(coefs, f)


if __name__ == '__main__':
    #visualize()
    main()