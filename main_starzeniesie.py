import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd
from nilearn import image, datasets
import nibabel as nib
from vedo import Volume, show, Mesh
import argparse
from pathlib import Path

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
        
        print("Kolumny", df)
        df['Norm_Coeff_age2'] = df['Norm_Coeff_age2'] * -1
        df = df.groupby('feature_name')['Norm_Coeff_age2'].sum().reset_index()
        self.importance_dict = dict(zip(df['feature_name'], df['Norm_Coeff_age2']))
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

    img=color_changer.read_atlas()
    color_changer.read_color_LUT()
    color_changer.read_feature_importance()
    importance_map=color_changer.map_feature_importance()


    #plotting.plot_roi(img, title='Atlas Regions', draw_cross=True, cmap='Set1'
    #plotting.show()

    new_img = nib.Nifti1Image(importance_map, affine=img.affine)
    vmin = np.min(importance_map)
    vmax = np.max(importance_map)

    '''
    display=plotting.plot_stat_map(new_img, cmap='rainbow', draw_cross=False, vmin=0, vmax=vmax)
    if display._cbar:
        display._cbar.ax.tick_params(labelsize=30)
    display.title('Feature Importance per Region', size=30)
    plotting.show()
    '''
    atlas_name = Path(args.atlas_file).name
    scores_name = Path(args.scores_file).stem

    out_file = f"plots/{atlas_name}_{scores_name}_brain_map.png"
    display=plotting.plot_glass_brain(new_img, cmap='rainbow', colorbar=True, vmin=0, vmax=vmax)
    if display._cbar:
        display._cbar.ax.tick_params(labelsize=30)
    display.title('Difference between healthy strucutres', size=30)
    display.savefig(out_file, dpi=300)
    plotting.show()

    #symulacja
    #color_changer.simulation(importance_map)


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
        default='brain_structures_hearing_transformed/starzenie_sie_struktur.csv',
        help="Path to the scores file (e.g., test_positive_importance_age_svm_valid_0_a2009_4.csv)"
    )
    #atlas_file='aparc.a2009saseg.nii.gz'
    #'aparcaseg.nii.gz'
    #scores_file='test_positive_importance_age_svm_valid_0_a2009.csv'
    #'test_positive_importance_age_svm_valid_0_ASEG_4.csv',
    #brain_structures_hearing_transformed/hearing_znaczny_trends_M_robust_scaled_degree_2_r2_HF.csv',
    parser.add_argument(
        '--reference_scores_file', 
        type=str, 
        default='brain_structures_hearing_transformed/hearing_gleboki_trends_M_robust_scaled_degree_2_r2_HF.csv', 
        help="Path to the scores file (e.g., test_positive_importance_age_svm_valid_0_ASEG.csv)"
    )
    args = parser.parse_args()
    main(args)


