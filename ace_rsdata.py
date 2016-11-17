# Note: you will need to run `source ENV/bin/activate` before running 
from __future__ import print_function
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


#NIFTI1 img
if __name__ == "__main__":
	import nibabel
	x = nibabel.load(get_list_nii_gz()[0])
	data = x.get_data()
	print(data.shape)
