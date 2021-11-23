import numpy as np
import sys
import os
import rw as rw


def read_grid(gridpath):
    dxC = rw.readfield(gridpath + 'DXC.bin',(3072,3360),'float32')
    dyC = rw.readfield(gridpath + 'DYC.bin',(3072,3360),'float32')
    dxG = rw.readfield(gridpath + 'DXG.bin',(3072,3360),'float32')
    dyG = rw.readfield(gridpath + 'DYG.bin',(3072,3360),'float32')
    return dxC,dyC,dyG,dxG
 
def read_latlon(gridpath):
    lon = rw.readfield(gridpath + 'LONC.bin',(3072,3360),'float32')
    lat = rw.readfield(gridpath + 'LATC.bin',(3072,3360),'float32')
    return lon,lat

def read_grid_1km(gridpath):
    dxC = rw.readfield(gridpath + 'DXC_4320x5280',(5280,4320),'float32')
    dyC = rw.readfield(gridpath + 'DYC_4320x5280',(5280,4320),'float32')
    dxG = rw.readfield(gridpath + 'DXG_4320x5280',(5280,4320),'float32')
    dyG = rw.readfield(gridpath + 'DYG_4320x5280',(5280,4320),'float32')
    return dxC,dyC,dyG,dxG
  
def read_latlon_1km(gridpath):
    lon = rw.readfield(gridpath + 'XC_4320x5280',(5280,4320),'float32')
    lat = rw.readfield(gridpath + 'YC_4320x5280',(5280,4320),'float32')
    return lon,lat
 

def read_output(fileU,fileV,fileA,is_itd=False):
    U = rw.readfield(fileU,(3072,3360),'float32')
    V = rw.readfield(fileV,(3072,3360),'float32')
    if is_itd:
        A = rw.readfield(fileA,(5,3072,3360),'float32')
        return U,V,np.sum(A,axis=0)
    else:
        A = rw.readfield(fileA,(3072,3360),'float32')
        return U,V,A


def read_output_1km(fileU,fileV,fileA):
    U = rw.readfield(fileU,(5280,4320),'float32')
    V = rw.readfield(fileV,(5280,4320),'float32')
    A = rw.readfield(fileA,(5280,4320),'float32')
    return U,V,A


def mask_arcticbasin(gridpath,read_func):
    lon,lat = read_func(gridpath)
    # lat = rw.readfield(gridpath + 'LATC.bin',(3072,3360),'float32')
    # lon = rw.readfield(gridpath + 'LONC.bin',(3072,3360),'float32')

    mask = ((((lon > -120) & (lon < 100)) & (lat >= 80)) |
            ((lon <= -120) & (lat >= 70)) |
            ((lon >= 100) & (lat >= 70)))

    return mask


def read_latlongrid(gridpath):
    lat = rw.readfield(gridpath + 'LATC.bin',(3072,3360),'float32')
    lon = rw.readfield(gridpath + 'LONC.bin',(3072,3360),'float32')
    return lon,lat


def grid_deformation_variables(dxC,dyC,dyG,dxG):
    # Compute help variables for grid deformation
    # dxF = 0.5*(dxC[1:-1,1:-1]+dxC[2:,1:-1])
    # dyF = 0.5*(dyC[1:-1,1:-1]+dyC[1:-1,2:])
    dxF = 0.5*(dxG[1:-1,1:-1]+dxG[1:-1,2:])
    dyF = 0.5*(dyG[1:-1,1:-1]+dyG[2:,1:-1])
    dyU = 0.5*(dyG[1:,:-1]+dyG[1:,1:])
    dxV = 0.5*(dxG[:-1,1:]+dxG[1:,1:])
    recip_dxF = 1/dxF
    recip_dyF = 1/dyF
    recip_dyU = 1/dyU
    recip_dxV = 1/dxV
    recip_dxF[np.isinf(recip_dxF)]=np.nan
    recip_dyF[np.isinf(recip_dyF)]=np.nan
    recip_dyU[np.isinf(recip_dyU)]=np.nan
    recip_dxV[np.isinf(recip_dxV)]=np.nan

    k1AtC = (dyG[2:,1:-1]-dyG[1:-1,1:-1])/(dxF*dyF)
    k2AtC = (dyG[1:-1,2:]-dyG[1:-1,1:-1])/(dxF*dyF)
    k1AtZ = (dyC[1:,1:]-dyC[:-1,1:])/(dyU*dxV)
    k2AtZ = (dxC[1:,1:]-dxC[1:,:-1])/(dxV*dyU)
    k1AtC[np.isinf(k1AtC)]=np.nan
    k2AtC[np.isinf(k2AtC)]=np.nan
    k1AtZ[np.isinf(k1AtZ)]=np.nan
    k2AtZ[np.isinf(k2AtZ)]=np.nan

    return recip_dxF,recip_dyF,recip_dyU,recip_dxV,k1AtC,k2AtC,k1AtZ,k2AtZ


def compute_deformation(U,V,A,recip_dxF,recip_dyF,recip_dyU,recip_dxV,k1AtC,k2AtC,k1AtZ,k2AtZ):

    # Filter low ice concentrations
    U[A == 0] = np.nan
    V[A == 0] = np.nan

    # Compute velocity derivatives at C-points:
    dudx = (U[2:,1:-1]-U[1:-1,1:-1])*recip_dxF
    uave = 0.5*(U[2:,1:-1]+U[1:-1,1:-1])
    dvdy = (V[1:-1,2:]-V[1:-1,1:-1])*recip_dyF
    vave = 0.5*(V[1:-1,2:]+V[1:-1,1:-1])
    #print('velocity derivatives computed')

    # Compute strainrates at C-points:
    e11 = dudx + vave*k2AtC
    e22 = dvdy + uave*k1AtC
    
    # Compute velocity derivatives at Z-points:
    dudy = (U[1:,1:]-U[1:,:-1])*recip_dyU
    uave = 0.5*(U[1:,1:]+U[1:,:-1])
    dvdx = (V[1:,1:]-V[:-1,1:])*recip_dxV
    vave = 0.5*(V[1:,1:]+V[:-1,1:])
    
    # Compute strainrates at Z-points:
    e12z = 0.5*(dudy+dvdx)-k1AtZ*vave-k2AtZ*uave
    
    # Average four Z-points on one C-point:
    e12c = 0.25*(e12z[1:,1:]+e12z[1:,:-1]+e12z[:-1,1:]+e12z[:-1,:-1])
    
    # Compute strainrate invariants:
    eps_I = (e11+e22) * 24. * 3600. # units from s^-1 to day ^-1
    eps_II = np.sqrt((e11-e22)**2+4*e12c**2) * 24. * 3600.
    eps_tot = np.sqrt(eps_I**2 + eps_II**2)
    
    return eps_tot, eps_I, eps_II


