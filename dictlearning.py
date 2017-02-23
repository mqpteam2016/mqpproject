from ace_rsdata import *
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func#get_list_nii_gz()


from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=5, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          random_state=0)
# Fit to the data
dict_learn.fit(func_filenames)
# Resting state networks/maps
components_img = dict_learn.masker_.inverse_transform(dict_learn.components_)

from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')

from nilearn.regions import RegionExtractor

extractor = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 5))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)
plotting.show()
datdat = get_list_nii_gz()
ts = []
for filename in datdat:
    timeseries_each_subject = extractor.transform(filename)
    ts.append(timeseries_each_subject)

import pickle

with open('outfile', 'wb') as fp:
    pickle.dump(ts, fp)
