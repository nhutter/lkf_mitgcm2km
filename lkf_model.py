import numpy as np
import os
import sys


# Self-written functions
from lkf_detection import *
from lkf_tracking import *
from model_utils import *



def lkf_detect_model(path_model_output,path_model_grid,path_processed,file_ind_start,file_ind_end,file_ind_period,max_kernel=5,min_kernel=1,dog_thres=5,dis_thres=4,ellp_fac=3,angle_thres=40,eps_thres=0.75,lmin=4,latlon=False,return_eps=False,is_itd=False):

    # Read model grid
    dxC,dyC,dyG,dxG = read_grid(path_model_grid)
    lon,lat = read_latlon(path_model_grid)
    # Compute grid deformation
    recip_dxF,recip_dyF,recip_dyU,recip_dxV,k1AtC,k2AtC,k1AtZ,k2AtZ = grid_deformation_variables(dxC,dyC,dyG,dxG)
    # Generate Arctic Basin mask
    mask = mask_arcticbasin(path_model_grid,read_latlon)
    index_x = np.where(np.sum(mask[1:-1,1:-1],axis=0)>0)
    index_y = np.where(np.sum(mask[1:-1,1:-1],axis=1)>0)
    red_fac = 3 # Take only every red_fac point to reduce array size

    # Look for all files
    filesU = [i for i in os.listdir(path_model_output) if i.startswith('SIuice') and
              i.endswith('.data') and int(i.split('.')[1]) >= file_ind_start and
              int(i.split('.')[1]) <= file_ind_end]
    filesV = [i for i in os.listdir(path_model_output) if i.startswith('SIvice') and
              i.endswith('.data') and int(i.split('.')[1]) >= file_ind_start and
              int(i.split('.')[1]) <= file_ind_end]
    filesA = [i for i in os.listdir(path_model_output) if i.startswith('SIarea') and
              i.endswith('.data') and int(i.split('.')[1]) >= file_ind_start and
              int(i.split('.')[1]) <= file_ind_end]
    filesU.sort(); filesV.sort(); filesA.sort()
    filesU = filesU[::file_ind_period]; filesV = filesV[::file_ind_period];
    filesA = filesA[::file_ind_period]

    for file_index in range(len(filesU)):
        print('Detection of file %i of %i' %(file_index+1, len(filesU)))
        # Read model data
        U,V,A = read_output(path_model_output + filesU[file_index],
                            path_model_output + filesV[file_index],
                            path_model_output + filesA[file_index],
                            is_itd=is_itd)
    
        # Compute deformation
        eps_tot, eps_I, eps_II = compute_deformation(U,V,A,recip_dxF,recip_dyF,recip_dyU,recip_dxV,
                                                     k1AtC,k2AtC,k1AtZ,k2AtZ)
        # Mask Arctic basin and shrink array
        eps_tot[~mask[1:-1,1:-1]] = np.nan
        eps_tot = eps_tot[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                          index_x[0][0]-1:index_x[0][-1]+2:red_fac]
        eps_tot[0,:] = np.nan; eps_tot[-1,:] = np.nan
        eps_tot[:,0] = np.nan; eps_tot[:,-1] = np.nan
        eps_tot[1,:] = np.nan; eps_tot[-2,:] = np.nan
        eps_tot[:,1] = np.nan; eps_tot[:,-2] = np.nan

        # Correct detection parameters for different resolution
        corfac = 12.5/2.246/float(red_fac)
    
        # Detect features
        lkf = lkf_detect_eps(eps_tot,max_kernel=max_kernel*corfac,
                             min_kernel=min_kernel*corfac,
                             dog_thres=dog_thres,dis_thres=dis_thres*corfac,
                             ellp_fac=ellp_fac,angle_thres=angle_thres,
                             eps_thres=eps_thres,lmin=lmin*corfac)
        
        # Save the detected features

        if latlon:
            lkf = segs2latlon_model(lkf,
                                    lon[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                        index_x[0][0]-1:index_x[0][-1]+2:red_fac],
                                    lat[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                        index_x[0][0]-1:index_x[0][-1]+2:red_fac])
        if return_eps:
            lkf =  segs2eps(lkf,
                            eps_I[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                  index_x[0][0]-1:index_x[0][-1]+2:red_fac],
                            eps_II[index_y[0][0]-1:index_y[0][-1]+2:red_fac,
                                   index_x[0][0]-1:index_x[0][-1]+2:red_fac])

        lkf_T = [j.T for j in lkf]
        np.save(path_processed + 'lkf_model_' + filesU[file_index].split('.')[1] + '.npy', lkf_T)

    return 'Done'

