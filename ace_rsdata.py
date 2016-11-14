# Note: you will need to run `source ENV/bin/activate` before running 
from __future__ import print_function
import pandas
import glob
import re

data_location = '/home/mgaskell/rsdata/'

# Gives us a dataframe size of 83x88 (rows 88-92 are Nans)
df = pandas.read_excel(data_location+'summary.xlsx')

# You can access the dataframe with things like df['SubjectID']


# Code for getting the zip files open.
# Doesn't work currently.
'''
def get_zipfilename(subject_id):
    if(not isinstance(subject_id, basestring)):
	print("Oops... subect_id is not a string")
	return []
    split_ids = re.split(r'[ ()]+', subject_id)[0:2]
    zip_files = []
    for id in split_ids:
	zip_files += (glob.glob(data_location + 'fmri/' + '*' + subject_id + '[^0-9]'))
    print(subject_id, len(zip_files))
    if(len(zip_files) == 0 and len(subject_id) != 4 and subject_id[1] == '0'):
	return get_zipfilename(subject_id[0] + subject_id[2:5])

for subject in df['SubjectID']:
    get_zipfilename(subject)
'''
