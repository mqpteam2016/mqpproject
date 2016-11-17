# Note: you will need to run `source ENV/bin/activate` before running 
from __future__ import print_function
import pandas
import re
import os

data_location = '/home/mgaskell/rsdata/'

# Gives us a dataframe size of 83x88 (rows 88-92 are Nans)
df = pandas.read_excel(data_location+'summary.xlsx')

# You can access the dataframe with things like df['SubjectID']

# Code for getting the zip files open.
# Doesn't work currently.

def get_zipfilename(subject_id):
    if(not isinstance(subject_id, basestring) and not isinstance (str(subject_id), basestring)):
	pprint.pprint(inspect.getmembers(subject_id))
	print("Oops... subect_id is not a string")
	return []
    if not hasattr(subject_id, '__len__'):
    	split_ids = [subject_id]
    else:
    	split_ids = [re.split(r'[ ()]+', str(x)) if str(x) else None for x in subject_id]
    zip_files = []
    for id in split_ids:
	zip_files += (glob.glob(data_location + 'fmri/' + '*' + str(subject_id) + '[^0-9]'))
    print(subject_id, len(zip_files))
    return zip_files

def get_list_nii_gz():
	gz_files = os.listdir(data_location + "nii_gz_files/")
	return [data_location+"nii_gz_files/"+ i for i in gz_files]


# Now we're going to match up the filenames to the value
#print(list(df['SubjectID']))
unmatched_rows = set(list(df['SubjectID']))
unmatched_files = set(get_list_nii_gz())
matched_files = {}
for filename in get_list_nii_gz():
	patient_id, date = re.split(r'[-/\.]', filename)[-4:-1][0:2]
	matches = filter(lambda id: (str(id).startswith(patient_id)), list(df['SubjectID']))
	# Found a match
	if not len(matches) == 0:
		if len(matches) > 1:
			print("File {:} matched multiple rows in the summary sheet: {:}".format(filename, matches)
		matched_files[filename] = matches[0] # Don't care about other matches
		for match in matches:
			unmatched_rows.discard(match)
		unmatched_files.remove(filename)

#NIFTI1 img
if __name__ == "__main__":
	import nibabel
	x = nibabel.load(get_list_nii_gz()[0])
	data = x.get_data()
	print(data.shape)
