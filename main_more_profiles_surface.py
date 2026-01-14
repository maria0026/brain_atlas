import nibabel as nib
from nilearn import plotting
import numpy as np
import nilearn
import pandas as pd
from nilearn import image, datasets
import nibabel as nib
from vedo import Volume, show, Mesh
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nilearn import plotting, surface, datasets

class ColorChanger():
    def __init__(self, atlas_file, LUT_file, scores_file):
        self.atlas_file = atlas_file
        self.LUT_file = LUT_file
        self.scores_file = scores_file


    def read_atlas(self):
        img = nib.load(self.atlas_file)
        self.data = img.get_fdata()
        self.unique_labels = np.unique(self.data)
        #print("Unikalne ID regionów:", self.unique_labels)
        return img


    def read_color_LUT(self):
        self.labels_dict = {}
        with open(self.LUT_file, 'r') as f:
            for line in f:
                #Skip comments
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        label_id = int(parts[0])
                        label_name = parts[1]
                        self.labels_dict[label_id] = label_name
            print(self.labels_dict)


    def read_feature_importance(self):
        df = pd.read_csv(self.scores_file, sep=None, engine="python")

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        #choose only relevant feature
        df = df[df['feature_name'].str.contains(args.feature)]

        df = df.applymap(
        lambda x: (x.replace('ASEG-', '')
                    .replace('_Volume_mm3', '')
                    .replace('_normMean', '')
                    .replace('_normStdDev', '')
                    .replace('_normMax', '')
                    .replace('_normMin', ''))
        if isinstance(x, str) else x
        )
        df = df.applymap(
        lambda x: (x.replace('A2009-', '')
                    .replace('_ThickStd', '')
                    .replace('_ThickAvg', '')
                    .replace('_GrayVol', '')
                    .replace('ctx-lh-', 'ctx_lh_')
                    .replace('ctx-rh-', 'ctx_rh_'))
        if isinstance(x, str) else x
        )
        df = df.groupby('feature_name')['quantile'].sum().reset_index()
        self.importance_dict = dict(zip(df['feature_name'], df['quantile']))
        print(self.importance_dict)
        return self.importance_dict
    

    def map_left_right_importance(self):
        importance_dict_left = {}
        importance_dict_right = {}
        for key in self.importance_dict.keys():
            if 'lh' in key:
                importance_dict_left[key] = self.importance_dict[key]
            elif 'rh' in key:
                importance_dict_right[key] = self.importance_dict[key]  
        return importance_dict_left, importance_dict_right
    
    
    def map_to_quantiles(self, importance_dict, quantiles):
        n_bins = len(quantiles) - 1  #8 klas
        for key, value in importance_dict.items():
            binned_value = np.digitize(value, quantiles, right=True)
            importance_dict[key] = n_bins + 1 - binned_value
        return importance_dict
    
    def read_atlas_surface(self):
        self.surface_atlas = datasets.fetch_atlas_surf_destrieux()
        labels = self.surface_atlas["labels"]  # lista nazw regionów
        self.surface_labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels]
        map_left = self.surface_atlas["map_left"]
        map_right = self.surface_atlas["map_right"]
        return map_left, map_right

    def create_texture(self, map, dict):
        texture = np.zeros_like(map, dtype=float)
        for idx, region_name in enumerate(self.surface_labels):
            if region_name in dict:
                texture[map == idx] = dict[region_name]
        texture = texture.astype(float)
        texture[texture == 0] = np.nan
        return texture


def fix_keys(dict):
    for key in list(dict.keys()):
        new_key = key.replace('ctx_lh_', '').replace('ctx_rh_', '')
        if new_key != key:
            dict[new_key] = dict.pop(key)
    return dict

def main(args):
    color_changer=ColorChanger(args.atlas_file, args.LUT_file, args.scores_file)

    img_org=color_changer.read_atlas()
    color_changer.read_color_LUT()
    importance_dict = color_changer.read_feature_importance()
    importance_dict_left, importance_dict_right = color_changer.map_left_right_importance()
    importance_dict_left = fix_keys(importance_dict_left)
    importance_dict_right = fix_keys(importance_dict_right)

    quantiles = np.array([0.1,1,5,25,50,75,95,99,100])
    #map to values from quantiles
    importance_dict_left = color_changer.map_to_quantiles(importance_dict_left, quantiles)
    importance_dict_right = color_changer.map_to_quantiles(importance_dict_right, quantiles)

    map_left, map_right = color_changer.read_atlas_surface()
    texture_left = color_changer.create_texture(map_left, importance_dict_left)
    texture_right = color_changer.create_texture(map_right, importance_dict_right)

    views=['lateral', 'medial', 'posterior', 'anterior']
    hemispheres = ['left', 'right']

    fig, axes = plt.subplots(
    nrows=2, ncols=4,
    subplot_kw={"projection": "3d"},
    figsize=(18, 9)
    )

    axes = axes.ravel()
    i=0
    fsavg = datasets.fetch_surf_fsaverage()

    for hemisphere in hemispheres:
        for view in views:
            if hemisphere == 'right':
                plotting.plot_surf_stat_map(
                    fsavg.infl_right,
                    texture_right,
                    hemi="right",
                    view=view,
                    cmap=plt.get_cmap('rainbow', len(quantiles)),
                    colorbar=True,
                    vmin=1,
                    vmax=len(quantiles) - 1,
                    axes=axes[i]
                )
            else:
                plotting.plot_surf_stat_map(
                    fsavg.infl_left,
                    texture_left,
                    hemi="left",
                    view=view,
                    cmap=plt.get_cmap('rainbow', len(quantiles)),
                    colorbar=True,
                    vmin=1,
                    vmax=len(quantiles) - 1,
                    axes=axes[i]
                )
            i+=1
    plt.show()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parser for atlas plotter")

    parser.add_argument(
        '--atlas_file', 
        type=str, 
        default='aparc.a2009saseg.nii.gz', 
        help="Path to the atlas file (e.g., aparcaseg.nii.gz)"
    )
    parser.add_argument(
        '--feature',
        type=str,
        default='ThickAvg',
        help="Feature to visualize (e.g., ThickAvg, Volume_mm3)"
    )
    parser.add_argument(
        '--LUT_file', 
        type=str, 
        default='FsTutorial_AnatomicalROI_FreeSurferColorLUT.txt', 
        help="Path to the LUT file (e.g., FsTutorial_AnatomicalROI_FreeSurferColorLUT.txt)"
    )
    parser.add_argument(
        '--scores_file', 
        type=str, 
        default='brain_structures_hearing_transformed/hearing_gleboki_trends_M_robust_scaled_degree_2_hf_False_sample_niedosluch_gleboki_2.csv',
        help="Path to the scores file (e.g., test_positive_importance_age_svm_valid_0_a2009_4.csv)"
    )
    #atlas_file='aparc.a2009saseg.nii.gz'
    #'aparcaseg.nii.gz'
    #scores_file='test_positive_importance_age_svm_valid_0_a2009.csv'
    #'test_positive_importance_age_svm_valid_0_ASEG_4.csv',
    args = parser.parse_args()
    main(args)


