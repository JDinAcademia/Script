#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:53:43 2022

Least square: is a standard approach in regression analysis to approximate the solution of overdetermined model
(e.g., equation in which there are more equation/knowns than unknowns) by minimizing the sum of the square of the residuals (error)
(the difference between an observed value and the fitted value provided by a model)

In this one, we use the LS-separate method to model each trial - every other things is similar to the LS-All one

Jan 24 2022 - add try except function to make it work for some subjs don't have all the data - adjust for the HPC sbatch 

@author: jd - Jacob -  Junqiang Dai; email: junqiang.dai@gmail.com; jdai@unc.edu
"""

# Set up the for loop fuction for analyzing each subject
# we have two separate shell files to call upon this script for each subject, they are:
    # The submission script (submit_job_array.sh) in longleaf for high-performance computating
    # The script that calls python for the subject (run_python_single_subject.sh) in longleaf for high-performance computating
# in order to run these sbatch files, they must be stored at the data directory under supercompute, teminal directory as well
import argparse
parser = argparse.ArgumentParser(description='calculating the similarity coefficients for each subject')
parser.add_argument('sub', metavar='Subject', type=str, nargs='+', help='Subject name')
args = parser.parse_args()
sub = args.sub[0]

# The example data is saved under 'Users/jd/nilearn_data/Self_postproc/func'

# STEP 1 - import the functioonal imaging data for each condition 
# import the fMRI dataset for self.
import os 
import numpy as np
import pandas as pd
from nilearn.image import load_img, coord_transform
try:
    data_path_self = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
    func_self_filename = os.path.join (data_path_self, sub+'_task-CupsSelf_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    confound_self_filename = os.path.join (data_path_self,sub+'_task-CupsSelf_desc-confounds_regressors.tsv')
    confound_self = pd.read_csv(confound_self_filename, delimiter='\t')
# only use frame-wise displacement as confound here
    confound_self_df = confound_self[['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']].fillna(0)
# from load_confounds import Minimal
# confounds = Minimal().load(file)
    confound_self_df.shape
    
    # check the file -3D or 4D for self
    import nibabel as nib
    func_self = nib.load (func_self_filename)
    func_self.shape
        #func_self = load_img(img_self.fun[0])
except Exception as e: print(e)

# import the fMRI dataset for parent. 
try:
    data_path_parent = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
    func_parent_filename = os.path.join(data_path_parent, sub+'_task-CupsParent_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    # check the file -3D or 4D for parent
    func_parent = nib.load(func_parent_filename)
    func_parent.shape
        #func_parent = load_img(img_parent.fun[0])
    confound_parent_filename = os.path.join (data_path_parent,sub+'_task-CupsParent_desc-confounds_regressors.tsv')
    confound_parent = pd.read_csv(confound_parent_filename, delimiter='\t')
#confound_parent_df = confound_parent[['trans_x','trans_x_derivative1','trans_y','trans_y_derivative1','trans_z','trans_z_derivative1']].fillna(0)
    confound_parent_df = confound_parent[['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']].fillna(0)
    confound_parent_df.shape
except Exception as e: print(e)

# import the fMRI dataset for peer. 
try:
    data_path_peer = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
    func_peer_filename = os.path.join(data_path_peer, sub+'_task-CupsPeer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    # check the file -3D or 4D for parent
    func_peer = nib.load(func_peer_filename)
    func_peer.shape
#func_peer = load_img(img_peer.fun[0])
    confound_peer_filename = os.path.join (data_path_peer, sub+'_task-CupsPeer_desc-confounds_regressors.tsv')
    confound_peer= pd.read_csv(confound_peer_filename, delimiter='\t')
#confound_peer_df = confound_peer[['trans_x','trans_x_derivative1','trans_y','trans_y_derivative1','trans_z','trans_z_derivative1']].fillna(0)
    confound_peer_df = confound_peer[['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']].fillna(0)
    confound_peer_df.shape
except Exception as e: print(e)

# STEP 2 - import the event time files for each condition and make LSA/LSS-compatable event files,

# use LSS to prepare the event files  
    # define lss_transformer function first 
def lss_transformer(df, row_number):
    """Label one trial for one LSS model."""
    df = df.copy()
    # Technically, all you need is for the requested trial to have a unique "trial_type" *within*
    # the dataframe, rather than across models.
    # However, we may want to have meaningful "trial_type"s (e.g., "Left_001") across models,
    # so that you could track individual trials across models.
    df["old_trial_type"] = df["trial_type"]
    # Determine which number trial it is *within the condition*.
    trial_condition = df.loc[row_number, "old_trial_type"]
    trial_type_series = df["old_trial_type"]
    trial_type_series = trial_type_series.loc[trial_type_series == trial_condition]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)
    trial_name = f"{trial_condition}_{trial_number:03d}"
    df.loc[row_number, "trial_type"] = trial_name
    
    return df, trial_name

# import the time files for each condition (self, parent, and peer)
import pandas as pd
onset_path_self = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
onset_path_parent = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
onset_path_peer = ('/proj/telzerlab/NT/data_fmriprep/Cups_RSA_JD_T1/'+sub+'/func/')
    # Self condition
try:    
    events_self_filename = os.path.join (onset_path_self, sub+'_CupsSelf_durationRT.tsv')
    events_self = pd.read_table(events_self_filename)
    condition_beta_maps_self = {cond: [] for cond in events_self["trial_type"].unique()}
except Exception as e: print(e)

    # Parent condition
try:
    events_parent_filename = os.path.join (onset_path_parent, sub+'_CupsParent_durationRT.tsv')
    events_parent = pd.read_table(events_parent_filename)
    condition_beta_maps_parent = {cond: [] for cond in events_parent["trial_type"].unique()}
except Exception as e: print(e)

    # Peer condition
try:
    events_peer_filename = os.path.join (onset_path_peer, sub+'_CupsPeer_durationRT.tsv')
    events_peer = pd.read_table(events_peer_filename)
    condition_beta_maps_peer = {cond: [] for cond in events_peer["trial_type"].unique()}
except Exception as e: print(e)


# STEP 3 - Conduct the GLMs for each trial in each condition 

TR = 2
# defined the ROI path - given that we have created the ROI using FSL, we only need to direct to the file
    # otherwise, we need to use NiftiMasker function - check other script  
masker_leftNACC = ('/proj/telzerlab/NT/data_glm/Cups_RSA/ROIs_Selection/ROI_leftNACC_7_7_6_sphere5_bin_resampled.nii.gz')
masker_rightNACC = ('/proj/telzerlab/NT/data_glm/Cups_RSA/ROIs_Selection/ROI_rightNACC_8_11_5_sphere5_bin_resampled.nii.gz')
masker_vmPFC = ('/proj/telzerlab/NT/data_glm/Cups_RSA/ROIs_Selection/harvardoxford-cortical_prob_FrontalMedial_resampled.nii.gz')
masker_leftTPJ = ('/proj/telzerlab/NT/data_glm/Cups_RSA/ROIs_Selection/ROI_leftTPJ_49_56_19_sphere5_bin_resampled.nii')
masker_rightTPJ = ('/proj/telzerlab/NT/data_glm/Cups_RSA/ROIs_Selection/ROI_rightTPJ_sphere5_bin_resampled.nii.nii')

        
from nilearn.glm.first_level import FirstLevelModel

        # Self
# for trial_condition in trialwise_conditions_self:
try:
    for i_row, trial_row in events_self.iterrows():
        lss_events_df_self, trial_condition_self = lss_transformer(events_self, i_row)
    # defined the GLM and fit the GLM (Note that the GLMs for all trials here are calculated in for loop)
    # so that you can collect the trial-wise beta map
        glm_self_lNACC = FirstLevelModel(t_r=2,slice_time_ref=0.5,
                          mask_img=masker_leftNACC,
                          high_pass=.008,
                          smoothing_fwhm=None,
                          memory='nilearn_cache')
        glm_self_lNACC.fit(func_self_filename,events = lss_events_df_self,confounds=confound_self_df)
    
        beta_map_self_lNACC = glm_self_lNACC.compute_contrast(trial_condition_self,output_type="effect_size")
    # Drop the trial number from the condition name to get the original name.
        condition_name_self = "_".join(trial_condition_self.split("_")[:-1])
    # condition_name = "_".join(condition)
        condition_beta_maps_self[condition_name_self].append(beta_map_self_lNACC)
except Exception as e: print(e)

        # Parent
try:
    for i_row, trial_row in events_parent.iterrows():
        lss_events_df_parent, trial_condition_parent = lss_transformer(events_parent, i_row)

        glm_parent_lNACC = FirstLevelModel(t_r=2,slice_time_ref=0.5,
                          mask_img=masker_leftNACC,
                          high_pass=.008,
                          smoothing_fwhm=None,
                          memory='nilearn_cache')
        glm_parent_lNACC.fit(func_parent_filename,events = lss_events_df_parent,confounds=confound_parent_df)
    
        beta_map_parent_lNACC = glm_parent_lNACC.compute_contrast(trial_condition_parent,output_type="effect_size")
        condition_name_parent = "_".join(trial_condition_parent.split("_")[:-1])
        condition_beta_maps_parent[condition_name_parent].append(beta_map_parent_lNACC)
except Exception as e: print(e)

        # Peer
try:
    for i_row, trial_row in events_peer.iterrows():
        lss_events_df_peer, trial_condition_peer = lss_transformer(events_peer, i_row)
        glm_peer_lNACC = FirstLevelModel(t_r=2,slice_time_ref=0.5,
                          mask_img=masker_leftNACC,
                          high_pass=.008,
                          smoothing_fwhm=None,
                          memory='nilearn_cache')
        glm_peer_lNACC.fit(func_peer_filename,events = lss_events_df_peer,confounds=confound_peer_df)
    
        beta_map_peer_lNACC = glm_peer_lNACC.compute_contrast(trial_condition_peer,output_type="effect_size")
        condition_name_peer = "_".join(trial_condition_peer.split("_")[:-1])
        condition_beta_maps_peer[condition_name_peer].append(beta_map_peer_lNACC)
except Exception as e: print(e)


# Validation check - Plot design matrix to check the new parameters
# the right dm should look like 45 single trial - e.g., peer_001...... peer_045
"""
from nilearn.glm.first_level import FirstLevelModel
first_level_model = FirstLevelModel(TR)
first_level_model = glm_peer_lNACC.fit (func_peer_filename, events = lsa_events_peer)
design_matrix = first_level_model.design_matrices_[0]
from nilearn.plotting import plot_design_matrix
plot_design_matrix(design_matrix)
import matplotlib.pyplot as plt
plt.show()
"""

# STEP 4 -  Get an array of all the betas for trials
from nilearn import masking

# Self-lNACC
try:
    condition_beta_self_lNACC_arrs = {}  
    for condition, maps in condition_beta_maps_self.items():
        beta_self_lNACC_arr = masking.apply_mask(maps, masker_leftNACC, dtype='f', smoothing_fwhm=None, ensure_finite=True)         
        condition_beta_self_lNACC_arrs[condition] = beta_self_lNACC_arr
# beta_arr = masking.apply_mask(imgs, masker)
# NOTE: The end result should be a dictionary of condition: (trial-by-voxel)
# numpy array pairs.
# The end result should be a dictionary of condition: (trial-by-voxel) numpy array pairs.
except Exception as e: print(e)

# Parent-lNACC
try:
    condition_beta_parent_lNACC_arrs = {}  
    for condition, maps in condition_beta_maps_parent.items():
        beta_parent_lNACC_arr = masking.apply_mask(maps, masker_leftNACC, dtype='f', smoothing_fwhm=None, ensure_finite=True)         
        condition_beta_parent_lNACC_arrs[condition] = beta_parent_lNACC_arr    
except Exception as e: print(e)

# Peer-lNACC
try:
    condition_beta_peer_lNACC_arrs = {}  
    for condition, maps in condition_beta_maps_peer.items():
        beta_peer_lNACC_arr = masking.apply_mask(maps, masker_leftNACC, dtype='f', smoothing_fwhm=None, ensure_finite=True)         
        condition_beta_peer_lNACC_arrs[condition] = beta_peer_lNACC_arr   
except Exception as e: print(e)

# STEP 5 - Calculate the similarity matrix
# there are three goals/ways to caclulate the similarity here
  # One - using multi-voxel method, whether decision making for self, parent, and peer are represented in ROIs
    # t-test comparing to 0
  # Two - whether the similarity among voxels in ROIs are different between self, parent, and peer
    # paired t-test 
  # Three - whether the neural response to self, parent, and peer are similarly represented across voxels in ROIs
    # e.g., self & parent, calculate the correlation between these two matrix at voxel level (Self_V1&Parent_V1....SV81&P81)
      # then average the similarity scores from voxels
        # compare to 0 using t-test; 
        # compare self&parent vs. self&peer using paired t-test
# One & Two
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# make a column name for the data array
features_lNACC = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
           "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
           "V21","V22","V23","V24","V25","V26","V27","V28","V29","V30",
           "V31","V32","V33","V34","V35","V36","V37","V38","V39","V40",
           "V41","V42","V43","V44","V45","V46","V47","V48","V49","V50",
           "V51","V52","V53","V54","V55","V56","V57","V58","V59","V60",
           "V61","V62","V63","V64","V65","V66","V67","V68","V69","V70",
           "V71","V72","V73","V74","V75","V76","V77","V78","V79","V80",
           "V81"]
# prepare data frames - add the feature for further analysis
try:
    df_self_lNACC = pd.DataFrame (beta_self_lNACC_arr, columns = features_lNACC)
    print (df_self_lNACC)
except Exception as e: print(e)
try:
    df_parent_lNACC = pd.DataFrame (beta_parent_lNACC_arr, columns = features_lNACC)
    print (df_self_lNACC)
except Exception as e: print(e)
try:
    df_peer_lNACC = pd.DataFrame (beta_peer_lNACC_arr, columns = features_lNACC)
    print (df_peer_lNACC)
except Exception as e: print(e)

# creat correlation matrics for self, parent, and peer
try:
    voxelcorr_self_lNACC = df_self_lNACC.corr (method='pearson', min_periods=1)
except: print('code did not work for this subject, continuing')
#plt.figure(figsize=(50,50))
#sns.heatmap (voxelcorr_self_lNACC, cmap='coolwarm', annot=True)
#plt.show()
try:
    voxelcorr_parent_lNACC = df_parent_lNACC.corr (method='pearson', min_periods=1)
except Exception as e: print(e)
#plt.figure(figsize=(50,50))
#sns.heatmap (voxelcorr_parent_lNACC, cmap='coolwarm', annot=True)
#plt.show()
try:
    voxelcorr_peer_lNACC = df_peer_lNACC.corr (method='pearson', min_periods=1)
except Exception as e: print(e)
#plt.figure(figsize=(50,50))
#sns.heatmap (voxelcorr_peer_lNACC, cmap='coolwarm', annot=True)
#plt.show()

# calculate the mean of coefficients between voxels for self, parent, and peer
try:
    voxelcorr_self_lNACC_nondiagonal = voxelcorr_self_lNACC.copy()
    voxelcorr_self_lNACC_nondiagonal.values[np.tril_indices_from(voxelcorr_self_lNACC_nondiagonal,1)] = np.nan
    voxelcorr_self_lNACC_nondiagonal
    voxelcorr_self_lNACC_nondiagonal.unstack().mean()
    voxelcorr_self_lNACC_similarity_mean = voxelcorr_self_lNACC_nondiagonal.unstack().mean()
    print (voxelcorr_self_lNACC_similarity_mean)
except Exception as e: 
    print(e) 
    voxelcorr_self_lNACC_similarity_mean = float ("NaN")

try:
    voxelcorr_parent_lNACC_nondiagonal = voxelcorr_parent_lNACC.copy()
    voxelcorr_parent_lNACC_nondiagonal.values[np.tril_indices_from(voxelcorr_parent_lNACC_nondiagonal,1)] = np.nan
    voxelcorr_parent_lNACC_nondiagonal
    voxelcorr_parent_lNACC_nondiagonal.unstack().mean()
    voxelcorr_parent_lNACC_similarity_mean = voxelcorr_parent_lNACC_nondiagonal.unstack().mean()
    print (voxelcorr_parent_lNACC_similarity_mean)
except Exception as e: 
    print(e)
    voxelcorr_parent_lNACC_similarity_mean = float ("NaN")

try:
    voxelcorr_peer_lNACC_nondiagonal = voxelcorr_peer_lNACC.copy()
    voxelcorr_peer_lNACC_nondiagonal.values[np.tril_indices_from(voxelcorr_peer_lNACC_nondiagonal,1)] = np.nan
    voxelcorr_peer_lNACC_nondiagonal
    voxelcorr_peer_lNACC_nondiagonal.unstack().mean()
    voxelcorr_peer_lNACC_similarity_mean = voxelcorr_peer_lNACC_nondiagonal.unstack().mean()
    print (voxelcorr_peer_lNACC_similarity_mean)
except Exception as e: 
    print(e) 
    voxelcorr_peer_lNACC_similarity_mean = float ("NaN")

# Three - 
# COMPUTE PAIRWISE CORRELATION OF COLUMNS (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html#pandas.DataFrame.corr).
    # axis = 0 means correlation between columns while 1 means corr between row?
    # similarity 1 - self & parent, similarity 2 - self & peer, similarity 3 - parent & peer
try:
    similarity1_lNACC = df_self_lNACC.corrwith(df_parent_lNACC, axis=0, method='pearson')
except Exception as e: 
    print(e)
    similarity1_lNACC = float ("NaN")
    
try:
    similarity2_lNACC = df_self_lNACC.corrwith(df_peer_lNACC, axis=0, method='pearson')
except Exception as e: 
    print(e)
    similarity2_lNACC = float ("NaN")
    
try:
    similarity3_lNACC = df_parent_lNACC.corrwith(df_peer_lNACC, axis=0, method='pearson')
except Exception as e: 
    print(e)
    similarity3_lNACC = float ("NaN")

try:
    similarity1_lNACC_coefficient = similarity1_lNACC.mean(axis=None, skipna=True)
except Exception as e: 
    print(e)
    similarity1_lNACC_coefficient = float ("NaN")
    
try:
    similarity2_lNACC_coefficient = similarity2_lNACC.mean(axis=None, skipna=True)
except Exception as e: 
    print(e)
    similarity2_lNACC_coefficient = float ("NaN")
    
try:
    similarity3_lNACC_coefficient = similarity3_lNACC.mean(axis=None, skipna=True)
except Exception as e: 
    print(e)
    similarity3_lNACC_coefficient = float ("NaN")

try:
    print (similarity1_lNACC_coefficient)
except Exception as e: print(e)

try:
    print (similarity2_lNACC_coefficient)
except Exception as e: print(e)

try:
    print (similarity3_lNACC_coefficient)
except Exception as e: print(e)

# STEP 6 - Group all data into a dataframe for one subject
    # check this pandas check list for more functions (https://www.studocu.com/en-au/document/university-of-new-south-wales/financial-market-data-design-and-analysis/pandas-cheatsheet-cheatsheet/9228558)
import pandas as pd
sub_data = {'voxelcorr_self_lNACC_similarity_mean':[voxelcorr_self_lNACC_similarity_mean],
               'voxelcorr_parent_lNACC_similarity_mean':[voxelcorr_parent_lNACC_similarity_mean],
               'voxelcorr_peer_lNACC_similarity_mean':[voxelcorr_peer_lNACC_similarity_mean],
               'similarity1_lNACC_coefficient':[similarity1_lNACC_coefficient],
               'similarity2_lNACC_coefficient': [similarity2_lNACC_coefficient],
               'similarity3_lNACC_coefficient':[similarity3_lNACC_coefficient]}

sub_RSA_df = pd.DataFrame(sub_data)
sub_RSA_df.to_csv ('/proj/telzerlab/NT/data_glm/Cups_RSA/Similarity_Coefficients_durationRT/T1/leftNACC/'+sub+'_RSA_leftNACC_LSS_effectsize.csv', index=False, header=True, sep =',')










