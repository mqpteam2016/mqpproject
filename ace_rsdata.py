# Note: you will need to run `source ENV/bin/activate` before running 
from __future__ import print_function

# Complains that scipy isn't installed if we don't do this...
# Because we don't have administrative access, we can't change /usr/lib64, and have to deal with what is already installed.
# Ideally, we'd just use our virtualenv, but sklearn needs to be installed in ~/.local so it *can* access scipy, which can't be installed in the venv because it relies of OpenBlas
import sys
sys.path.append("/usr/lib64/python2.7/site-packages")
sys.path.append("/home/mgaskell/.local/lib/python2.7/site-packages")
sys.path.append("/home/mgaskell/work/mqpproject/ENV/lib64/python2.7")
import pandas
import re
import os

data_location = '/home/mgaskell/rsdata/'

# Gives us a dataframe size of 83x88 (rows 88-92 are Nans)
df = pandas.read_excel(data_location+'summary.xlsx')

# You can access the dataframe with things like df['SubjectID']

def get_list_nii_gz():
	gz_files = os.listdir(data_location + "nii_gz_files/")
	return [data_location+"nii_gz_files/"+ i for i in gz_files]


# Now we're going to match up the filenames to the value
unmatched_rows = set(list(df['SubjectID']))
unmatched_files = set(get_list_nii_gz())
matched_files = {}
for filename in get_list_nii_gz():
	patient_id, date = re.split(r'[-/\.]', filename)[-4:-1][0:2]
	matches = filter(lambda id: (str(id).startswith(patient_id)), list(df['SubjectID']))
	# Found a match
	if not len(matches) == 0:
		if len(matches) > 1:
			print("File {:} matched multiple rows in the summary sheet: {:}".format(filename, matches))
		matched_files[filename] = matches[0] # Don't care about other matches
		for match in matches:
			unmatched_rows.discard(match)
		unmatched_files.remove(filename)


def get_dataset(y_column_name='Age'):
	import nibabel
	filenames = sorted(matched_files.keys())
	Y = [ df.loc[df['SubjectID'] == matched_files[filename]][y_column_name] for filename in filenames]
	X = [ nibabel.load(filename) for filename in filenames]
	return X, Y
# X[0].get_data() has shape (96, 96, 50, 156), so you might want to reorder them with
# np.moveaxis(X[0].get_data(), -1, 0)

#NIFTI1 img
if __name__ == "__main__":
	import nibabel
	x = nibabel.load(get_list_nii_gz()[0])
	data = x.get_data()
	print(data.shape)
