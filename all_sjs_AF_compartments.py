# coding=utf-8


import pandas as pd



import nibabel as nib
# from nibabel.processing import resample_to_output
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes

# import nilearn as nil

import numpy as np

import dipy

from dipy.segment.clustering import QuickBundles

from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from dipy.tracking import utils

import os


Tractogram = nib.streamlines.tractogram.Tractogram

ArraySequence = nib.streamlines.array_sequence.ArraySequence



#%%

dir_atlas = '/mnt/Dat1/Z_Atlases_in_subject_space---'


codes = pd.read_csv(dir_atlas + '/' + 'ROI_AF_comp.m', header=None, names=['reg', 'regno'], sep=' ')

ROIs = list( codes['reg'] )

ROI_codes = list( codes['regno'] )


# %%

dir_top = '/mnt/DAT7/Z_TRACTOGRAMS_AF_filt'

dir_top_o = '/mnt/DAT7/Z_TRACTOGRAMS_AF_filt'

# os.mkdir(dir_top_o )

#%%

import glob

sjs_dirs = glob.glob( dir_top + '/' +  '1*_*' , recursive=False  )

sjs = [  sjs_dirs[ii].split('/')[-1] for ii in range(len(sjs_dirs)) ]

sjs = sorted(sjs)





#%%

# sj = '150617_CP_s2_150902_CP_large_ftp_lh__post'
#
from joblib import Parallel, delayed

for sj in sjs:

    dir_sj = dir_top + '/' + sj

    dir_sj_o = dir_top_o + '/' + sj
    # os.mkdir( dir_sj_o )

    dir_atlas_sj = dir_atlas + '/' + sj

    atlas = dir_atlas_sj + '/JHU289_modified_t1.nii.gz'

    img_atlas = nib.load(atlas)

    atl = img_atlas.get_data()# feature = ResampleFeature(nb_points=50)
    # metric = AveragePointwiseEuclideanMetric(feature=feature)
    # qb = QuickBundles(threshold=5, metric=metric)
    # clusters = qb.cluster(streamlines)
    #
    # print(len(streamlines))
    # print(clusters.clusters_sizes())
    #
    # clusters.remove_cluster(*clusters.get_small_clusters(2))
    #
    # print(clusters.clusters_sizes())
    #
    # idx = []
    # for ii in range(len(clusters)):
    #     idx += clusters[ii].indices
    #
    # streamlines = streamlines[idx]
    # print(len(streamlines))


    codes = pd.read_csv(dir_atlas_sj + '/' + 'JHU289_modified_labels_codes.txt', header=None, names=['reg', 'regno'], sep=' ')


    rois = np.zeros_like( atl )

    for ix, r in enumerate( ROIs ):
        roi = np.zeros_like( atl )
        roi[ np.where( atl == ROI_codes[ix]) ] = ROI_codes[ix]
        # img_roi = nib.Nifti1Image( roi , img_atlas.affine)
        # img_roi.to_filename( dir_sj_o + '/' +  '/' 'roi_AF_' +  ROIs[ix] + '.nii.gz' )
        rois[ np.where( atl == ROI_codes[ix]) ] = ROI_codes[ix]



    # img_roi = nib.Nifti1Image( roi , img_atlas.affine)
    #
    #
    # img_roi.to_filename( dir_sj_o + '/' + 'ROIs_AF.nii.gz' )

#%%



for sj in sjs:

    dir_sj = dir_top + '/' + sj

    dir_sj_o = dir_top_o + '/' + sj
    # os.mkdir( dir_sj_o )

    dir_atlas_sj = dir_atlas + '/' + sj

    atlas = dir_atlas_sj + '/JHU289_modified_t1.nii.gz'

    img_atlas = nib.load(atlas)

    atl = img_atlas.get_data()

    codes = pd.read_csv(dir_atlas_sj + '/' + 'JHU289_modified_labels_codes.txt', header=None, names=['reg', 'regno'], sep=' ')


    rois = np.zeros_like( atl )

    for ix, r in enumerate( ROIs ):
        rois[np.where(atl == ROI_codes[ix])] = ix

    img_roi = nib.Nifti1Image( rois , img_atlas.affine)
    img_roi.to_filename( dir_sj_o + '/' +  '/' 'roi_AF_' +  'sorted.nii.gz' )








#%%

for sj in sjs:

    #%%

    dir_sj = dir_top_o + '/' + sj

    dir_sj_o = dir_top_o + '/' + sj



    tck_AF =  nib.streamlines.load(  dir_sj + '/' +  'tck_AF_filt.tck' )

    streamlines = tck_AF.streamlines

    img_ROIs = nib.load( dir_sj + '/' +  'ROIs_AF.nii.gz' )

    labels =  img_ROIs.get_data().astype(int)



    affine = img_ROIs.affine


    M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)



    #%%


    streamlines_in = TCT_in.streamlines

    Tractogram = nib.streamlines.tractogram.Tractogram

    ArraySequence = nib.streamlines.array_sequence.ArraySequence

    streamlines = streamlines_in

    for kk in range(4):
        feature = ResampleFeature(nb_points=50)
        metric = AveragePointwiseEuclideanMetric(feature=feature)
        qb = QuickBundles(threshold=7.5, metric=metric)
        # qb = QuickBundles(threshold=10.)
        clusters = qb.cluster(streamlines)

        # print(len(streamlines))
        # print(clusters.clusters_sizes())

        clusters.remove_cluster(*clusters.get_small_clusters(5))

        # print(clusters.clusters_sizes())

        idx = []
        for ii in range(len(clusters)):
            idx += clusters[ii].indices

        streamlines = streamlines[idx]
        print(len(streamlines))

    # %%

    img_T1 = nib.load(dir_sj + '/T1.nii.gz')

    img_T1.to_filename(dir_sj_o + '/T1.nii.gz' )

    from dipy.align.reslice import reslice

    T1_2, affine_2 = reslice(img_T1.get_data(), img_T1.affine, img_T1.header.get_zooms(), (5, 5, 5))

    img_T1_2 = nib.Nifti1Image(T1_2, affine_2)

    # img_T1_2.to_filename(af_dir + '/T1__8mm.nii.gz')

    tdi = utils.density_map(streamlines, img_T1_2.shape, affine=img_T1_2.affine)

    tdi_img = nib.Nifti1Image(tdi.astype("int16"), img_T1_2.affine)
    # tdi_img.to_filename(af_dir + '/TDI_8mm.nii.gz')

    tdi_mask = np.zeros(tdi_img.get_data().shape)
    tdi_mask[np.where(tdi_img.get_data() > 1)] = 1

    tdi_mask_exc = np.zeros(tdi_img.get_data().shape)
    tdi_mask_exc[np.where(tdi_img.get_data() == 1)] = 1

    np.unique(tdi[np.where(tdi_mask > 0)])

    # tdi_img.get_data() > 1;


    #  np.unique(tdi[np.where(tdi_mask>0)])

    stl_keep = dipy.tracking.utils.target(streamlines, tdi_mask_exc, affine=tdi_img.affine, include=False)
    # stl_exc  = dipy.tracking.utils.target(streamlines, tdi_mask_exc, affine=tdi_img.affine, include=True)


    stl_keep = ArraySequence(stl_keep)
    # stl_exc = list(stl_exc)


    streamlines = stl_keep

    # feature = ResampleFeature(nb_points=50)
    # metric = AveragePointwiseEuclideanMetric(feature=feature)
    # qb = QuickBundles(threshold=5, metric=metric)
    # clusters = qb.cluster(streamlines)
    #
    # print(len(streamlines))
    # print(clusters.clusters_sizes())
    #
    # clusters.remove_cluster(*clusters.get_small_clusters(2))
    #
    # print(clusters.clusters_sizes())
    #
    # idx = []
    # for ii in range(len(clusters)):
    #     idx += clusters[ii].indices
    #
    # streamlines = streamlines[idx]
    # print(len(streamlines))


    tct_keep =  Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    # tct_exc = Tractogram(stl_exc, affine_to_rasmm=np.eye(4))

    nib.streamlines.save(tct_keep, dir_sj_o + '/' + 'tck_AF_filt.tck')
    # nib.streamlines.save(tct_exc, 'tck_Exc.tck')

    nii = img_T1
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    nib.streamlines.save(tct_keep, dir_sj_o + '/' + 'trk_AF_filt.trk', header=header)
    # nib.streamlines.save(tct_exc,  'trk_Exc.trk', header=header)

    tdi = utils.density_map(tct_keep.streamlines, img_T1.shape, affine=img_T1.affine)

    tdi_img = nib.Nifti1Image(tdi.astype("int16"), img_T1.affine)
    tdi_img.to_filename(dir_sj_o + '/TDI_AF_1mm.nii.gz')

    #%%













