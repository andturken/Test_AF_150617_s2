#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

# %%

import nibabel as nib
# from nibabel.processing import resample_to_output
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes

# import nilearn as nil

import numpy as np

import dipy

from dipy.segment.clustering import QuickBundles

# from dipy.segment.metric import ResampleFeature
# from dipy.segment.metric import AveragePointwiseEuclideanMetric

from dipy.tracking import utils

# import itertools
#
# import os
#
# import scipy.stats as stats

# %%

Tractogram = nib.streamlines.tractogram.Tractogram

ArraySequence = nib.streamlines.array_sequence.ArraySequence

# %%

'''
%run ~/Software/anaconda3/ipython_startup__import_common.py

%cd /mnt/DAT7/Z_TRACTOGRAMS_AF/150617_CP_s2_150902_CP_large_ftp_lh__post/tck_AF_all_Temp_Fr_Ins_RH
'''

# %%


sj_dir = '/mnt/DAT7/Z_TRACTOGRAMS_AF/150617_CP_s2_150902_CP_large_ftp_lh__post/'

af_dir = sj_dir + 'tck_AF_all_Temp_Fr_Ins_RH/'

fn = af_dir + 'tck_AF_all_Temp_Fr_Ins_RH_00.tck'

ftck = nib.streamlines.load(fn)

tct = nib.streamlines.load(fn).tractogram

idx_exc = set(range(len(ftck.streamlines))).difference(set(idx))

# %%

streamlines = ftck.streamlines

for kk in range(3):
    # feature = ResampleFeature(nb_points=20)
    # metric = AveragePointwiseEuclideanMetric(feature=feature)
    # qb = QuickBundles(threshold=10., metric=metric)
    qb = QuickBundles(threshold=10.)
    clusters = qb.cluster(streamlines)

    print(len(streamlines))
    print(clusters.clusters_sizes())

    clusters.remove_cluster(*clusters.get_small_clusters(9))

    print(clusters.clusters_sizes())

    idx = []
    for ii in range(len(clusters)):
        idx += clusters[ii].indices

    streamlines = streamlines[idx]
    print(len(streamlines))

# %%

# tck_out = nib.streamlines.tractogram.Tractogram( streamlines )

tck_out = ftck
tck_out.streamlines = streamlines

fn_o = af_dir + '/filtered.tck'
nib.streamlines.save(tck_out, fn_o)

# streamlines_2 =nib.streamlines.array_sequence.ArraySequence()
# streamlines_2.extend(clusters)

# for ii in range( len(clusters)):
#     streamlines_2.append( streamlines(clusters[ii].indices))


# n_clusters = clusters.size()
#
# csize = np.array( clusters.clusters_sizes() )
#
# csize_std = stats.zscore( csize )
#
# ix_single = np.where( csize == 1)
#
# clust_sz3 = clusters.get_small_clusters(3)
#
# clusters2 = clusters
#


# %% Make density map
#
# T1 = nib.load(sj_dir + '/T1.nii.gz')
#
# # T1.header
#
# dims = T1.header.get_data_shape()
# # T1.header.get_zooms()
# aff = T1.header.get_qform()
# dims = T1.shape
# aff = T1.affine
#
# T1_2mm = nil.image.resample_img(T1, target_affine=np.diag([2, 2, 2]))
# # T2.to_filename( af_dir + '/T1_2mm.nii.gz')
#
# tdi = utils.density_map(streamlines, T1_2mm.shape, affine=T1_2mm.affine)
#
# tdi_img = nib.Nifti1Image(tdi.astype("int16"), T1_2mm.affine)
# tdi_img.to_filename(af_dir + '/TDI_2mm.nii.gz')
#
# # tdi_shape = (np.array(dims)/2).round().astype(np.int)
# # tdi = utils.density_map( stl, vol_dims=tdi_shape  , voxel_size=2, affine=T1.affine )
#
# tdi_mask = np.zeros(tdi_img.get_data().shape)
#
# tdi_mask[np.where(tdi_img.get_data() > 1)] = 1
#
# # tdi_img.get_data() > 1;
#
#
#
#
# stl_inc = dipy.tracking.utils.target(streamlines, tdi_mask, affine=tdi_img.affine, include=True)
#
# stl_exc = dipy.tracking.utils.target(streamlines, tdi_mask, affine=tdi_img.affine, include=False)
#
# stl_inc

# %%

T1 = nib.load(sj_dir + '/T1.nii.gz')

tdi = utils.density_map(streamlines, T1.shape, affine=T1.affine)

tdi_img = nib.Nifti1Image(tdi.astype("int16"), T1.affine)
tdi_img.to_filename(af_dir + '/TDI_1mm.nii.gz')

from dipy.align.reslice import reslice

T1_2, affine_2 = reslice(T1.get_data(), T1.affine, T1.header.get_zooms(), (8, 8, 8))

img_T1_2 = nib.Nifti1Image(T1_2, affine_2)

img_T1_2.to_filename(af_dir + '/T1__8mm.nii.gz')

tdi = utils.density_map(streamlines, img_T1_2.shape, affine=img_T1_2.affine)

tdi_img = nib.Nifti1Image(tdi.astype("int16"), img_T1_2.affine)
tdi_img.to_filename(af_dir + '/TDI_8mm.nii.gz')

tdi_mask = np.zeros(tdi_img.get_data().shape)
tdi_mask[np.where(tdi_img.get_data() > 1)] = 1

tdi_mask_exc = np.zeros(tdi_img.get_data().shape)
tdi_mask_exc[np.where(tdi_img.get_data() == 1)] = 1

np.unique(tdi[np.where(tdi_mask > 0)])

# tdi_img.get_data() > 1;


#  np.unique(tdi[np.where(tdi_mask>0)])

stl_keep = dipy.tracking.utils.target(streamlines, tdi_mask_exc, affine=tdi_img.affine, include=False)
stl_exc  = dipy.tracking.utils.target(streamlines, tdi_mask_exc, affine=tdi_img.affine, include=True)

# stl_inc = dipy.tracking.utils.target(streamlines, tdi_mask, affine=tdi_img.affine, include=True)
# stl_exc = dipy.tracking.utils.target(streamlines, tdi_mask, affine=tdi_img.affine, include=False)

# len(list(stl_exc)), len(list(stl_keep))

stl_keep = list(stl_keep)
stl_exc = list(stl_exc)

tct_keep = Tractogram(stl_keep, affine_to_rasmm=np.eye(4))
tct_exc = Tractogram(stl_exc, affine_to_rasmm=np.eye(4))

nib.streamlines.save(tct_keep, 'tck_Keep.tck')
nib.streamlines.save(tct_exc, 'tck_Exc.tck')

nii = T1
header = {}
header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
header[Field.DIMENSIONS] = nii.shape[:3]
header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

nib.streamlines.save(tct_keep, 'trk_Keep.trk', header=header)
nib.streamlines.save(tct_exc,  'trk_Exc.trk',  header=header)

tct_0 = ftck.tractogram
nib.streamlines.save(tct_0, 'trk_All.trk', header=header)

tct_qb = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
nib.streamlines.save(tct_qb, 'trk_filtQB.trk', header=header)

tct_qb = Tractogram(ftck.streamlines[list(idx_exc)], affine_to_rasmm=np.eye(4))
nib.streamlines.save(tct_qb, 'trk_filtQB_Exc.trk', header=header)

# %%

stl_exc_qb = ftck.streamlines[list(idx_exc)]

STL = ArraySequence()
STL.extend( stl_keep )
STL.extend( stl_exc )
STL.extend( stl_exc_qb )

data_per_streamline = {'color': [] }

# colors_keep    = np.tile( np.array( [(0, 0, 1)]), (len( stl_keep) , 1 ) )
# colors_exc     = np.tile( np.array( [(0, 1, 0)]), (len( stl_exc) , 1 ) )
# colors_exc_qb  = np.tile( np.array( [(1, 0, 0)]), (len( stl_exc_qb) , 1 ) )
#
# colors = np.concatenate( [colors_keep, colors_exc, colors_exc_qb  ] , axis=0 )
#

# colors_keep    = np.tile( np.array( [0, 0, 1], dtype="f4"), (len( stl_keep) , 1 ) )
# colors_exc     = np.tile( np.array( [0, 1, 0], dtype="f4"), (len( stl_exc) , 1 ) )
# colors_exc_qb  = np.tile( np.array( [1, 0, 0], dtype="f4"), (len( stl_exc_qb) , 1 ) )
# colors = [ np.concatenate( [colors_keep, colors_exc, colors_exc_qb  ] , axis=0 ) ]


colors_keep    = [ np.array( [0, 0, 1], dtype="f4")  for ii in range( len( stl_keep) ) ]
colors_exc     = [ np.array( [0, 1, 0], dtype="f4")  for ii in range( len( stl_exc) ) ]
colors_exc_qb  = [ np.array( [1, 0, 0], dtype="f4")  for ii in range( len( stl_exc_qb) ) ]

colors = colors_keep + colors_exc  + colors_exc_qb

data_per_streamline = {'color': colors }



TCT = Tractogram( STL, data_per_streamline, affine_to_rasmm=np.eye(4) )


nib.streamlines.save( TCT , 'trk_All_Colored.trk', header=header)





#%%

from dipy.viz import window, actor

from dipy.utils.optpkg import optional_package
vtk, have_vtk, setup_module = optional_package('vtk')



