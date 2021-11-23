import numpy as np
import matplotlib.pylab as plt
import os
import sys
import datetime as dt

# Self-written functions
sys.path.append('path_to_lkf_tools_repo')
from lkf_detection import *
from lkf_tracking import *
from model_utils import *
from lkf_model_utils import *


# Options
rgps_sampling = False  # Samples RGPS dates and generates comparable data set
complete_data = True   # Samples all available model data


# Input data
path_model_grid   = '/work/ollie/nhutter/arctic_2km/run_itd_cor_cs/' # path to the binary grid files
path_model_output = '/work/ollie/nhutter/arctic_2km/output_damien/'  # path to the binary output files
model_start = dt.datetime(1992,1,1,0,0,0)
ts = 120.

output_path = '/work/ollie/nhutter/lkf_data/mitgcm_2km_damien/'

# Detectting parameters (parameters from Hutter et al., 2019, commented old version)
max_kernel   = 5    # 5
min_kernel   = 1    # 1
dog_thres    = 15   # 5
dis_thres    = 4    # 5
ellp_fac     = 2    # 3
angle_thres  = 35   # 45
eps_thres    = 1.25 # 0.75
lmin         = 3    # 4
latlon       = True
return_eps   = True
is_itd       = False


if rgps_sampling:
    years = [#'w0001',  'w0102',  'w0203',  'w0304',  
             #'w0405',  'w0506',  
             #'w0607',  'w0708']#,
             #'w9697',  'w9798',  'w9899',  'w9900']
             'w9798']

    RGPS_path = '/work/ollie/nhutter/RGPS/eulerian/'
    ts_RGPS = 3*24.*3600.


    # Iterate over years

    for year_dic in years:
        print "Start with season: " + year_dic
        new_dir = output_path + year_dic + '/'
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        # Check for RGPS dates
        file_list = os.listdir(RGPS_path + year_dic + '/')
        file_list.sort()
        start = (dt.datetime(int(file_list[0][:4]),1,1,0,0,0) + 
                 dt.timedelta(int(file_list[0][4:7])))
        end   = (dt.datetime(int(file_list[-1][-11:-7]),1,1,0,0,0) + 
                 dt.timedelta(int(file_list[-1][-7:-4])))
        
        lkf_detect_model(path_model_output,path_model_grid,new_dir,
                         int((start-model_start).total_seconds()/ts),
                         int((end-model_start).total_seconds()/ts),3,
                         max_kernel=max_kernel,min_kernel=min_kernel,
                         dog_thres=dog_thres,dis_thres=dis_thres,
                         ellp_fac=ellp_fac,angle_thres=angle_thres,
                         eps_thres=eps_thres,lmin=lmin,
                         latlon=latlon,return_eps=return_eps,
                         is_itd=is_itd)
    
    
if complete_data:
    new_dir = output_path[:-1] + '_complete_dataset/'
    if not os.path.exists(new_dir):
            os.mkdir(new_dir)
    # Check previous progress
    prog_file = 'prog_dataset_model.npz'

    par_dic = {'max_kernel'  : max_kernel,
               'min_kernel'  : min_kernel,
               'dog_thres'   : dog_thres,
               'dis_thres'   : dis_thres,
               'ellp_fac'    : ellp_fac,
               'angle_thres' : angle_thres,
               'eps_thres'   : eps_thres,
               'lmin'        : lmin}

    output_files = [ifile for ifile in os.listdir(path_model_output) if ifile.startswith('SIarea') and ifile.endswith('.data')]
    output_files.sort()

    dt_start = int(output_files[0].split('.')[1])
    dt_end   = int(output_files[-1].split('.')[1])
    
    num_files_it = 7200

    dt_i = dt_start

    while dt_i+num_files_it < dt_end:
        if os.path.exists(prog_file):
            prog_data = np.load(prog_file)
            
            if prog_data['par_dic'] == par_dic:
                dt_i = prog_data['dt_i']
            else:
                print("Error detection with different parameters exists: Break in order to change output directory")
                break

        print("New cycle of detection with saving intermediate iteration number: %i" %dt_i)


        lkf_detect_model(path_model_output,path_model_grid,new_dir,
                         dt_i, dt_i + num_files_it, 1,
                         max_kernel=max_kernel,min_kernel=min_kernel,
                         dog_thres=dog_thres,dis_thres=dis_thres,
                         ellp_fac=ellp_fac,angle_thres=angle_thres,
                         eps_thres=eps_thres,lmin=lmin,
                         latlon=latlon,return_eps=return_eps,
                         is_itd=is_itd)

        dt_i += num_files_it
        
        np.savez(prog_file,dt_i=dt_i,par_dic=par_dic)
        np.savez(new_dir + 'parameters.npz',par_dic=par_dic)
    

    
