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


    def get_feature_names(self):
        print("\nRegiony w atlasie:")
        for label_id in self.unique_labels:
            label_name = self.labels_dict.get(label_id, 'Unknown')
            print(f'ID {label_id}: {label_name}')

    def map_feature_importance(self):
        importance_map = np.zeros_like(self.data)

        for label_id in self.unique_labels:
            label_name = self.labels_dict.get(label_id, None)
            if label_name in self.importance_dict:
                importance_value = self.importance_dict[label_name]
                importance_map[self.data == label_id] = importance_value  # Wypełniamy voxele tej struktury
        return importance_map

    def simulation(self, importance_map):
        img = nib.load(self.atlas_file)
        data = img.get_fdata()
        volume_data = Volume(data)
        importance_volume = Volume(importance_map)
        importance_volume.cmap('hot', vmin=importance_map.min(), vmax=importance_map.max())
        show(volume_data, importance_volume, axes=1, title='Feature Importance 3D Visualization')

def main(args):
    color_changer=ColorChanger(args.atlas_file, args.LUT_file, args.scores_file)

    img_org=color_changer.read_atlas()
    color_changer.read_color_LUT()
    color_changer.read_feature_importance()
    importance_map=color_changer.map_feature_importance()

    quantiles = np.array([0.1,1,5,25,50,75,95,99,100])
    importance_map = np.digitize(importance_map, quantiles, right=True)
    importance_map = importance_map.astype(float)
    importance_map[importance_map == 0] = np.nan

    n_bins = len(quantiles) - 1  # 8 klas
    importance_map_transformed = n_bins + 1 - importance_map  # teraz 99–100 → 1, 0–1 → 8

    img = nib.Nifti1Image(importance_map, affine=img_org.affine)
    img_transformed = nib.Nifti1Image(importance_map_transformed, affine=img_org.affine)
    cmap = plt.get_cmap('rainbow', len(quantiles))
    cmap.set_under('white')

    atlas_name = Path(args.atlas_file).name
    scores_name = Path(args.scores_file).stem

    out_file = f"plots/{atlas_name}_{scores_name}_brain_map.png"
    display = plotting.plot_glass_brain(
        img_transformed,
        cmap=cmap,
        colorbar=True,
        vmin=1,
        vmax=len(quantiles) - 1
    )
    labels = ["99-100", "95-99", "75-95", "50-75", "25-50", "5-25", "1-5", "0-1"]
    if display._cbar:
        display._cbar.set_ticks(np.arange(1, len(quantiles)))
        display._cbar.set_ticklabels(labels)
    display.title('Difference between healthy structures', size=30)
    display.savefig(out_file, dpi=300 )
    plotting.show()

    display = plotting.plot_glass_brain(
        img,
        cmap=cmap,
        colorbar=True,
        vmin=1,
        vmax=len(quantiles) - 1
    )

    labels = [
    "0-1>", "1-5>", "5-25>", "25-50>",
    "50-75>", "75-95>", "95-99>", "99-100>"
    ]
    if display._cbar:
        display._cbar.set_ticks(np.arange(1, len(quantiles)))
        display._cbar.set_ticklabels(labels)
    display.title('Difference between healthy structures', size=30)
    display.savefig(out_file, dpi=300 )
    plotting.show()

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
        default='brain_structures_hearing_transformed/hearing_gleboki_trends_M_robust_scaled_degree_2_hf_False_sample_niedosluch_gleboki_6.csv',
        help="Path to the scores file (e.g., test_positive_importance_age_svm_valid_0_a2009_4.csv)"
    )
    #atlas_file='aparc.a2009saseg.nii.gz'
    #'aparcaseg.nii.gz'
    #scores_file='test_positive_importance_age_svm_valid_0_a2009.csv'
    #'test_positive_importance_age_svm_valid_0_ASEG_4.csv',
    args = parser.parse_args()
    main(args)


