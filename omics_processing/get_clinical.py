import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')


## RUN JUST ONE TIME
def get_clinical():

	# define dirs
	maindir = '/home/ubuntu/my-git-repos/netDL/'
	datadir = maindir+'DATA/TCGA_fromMatteo/all_processed/'
	datadir_out = maindir+'DATA/TCGA_fromMatteo/all_processed/formatted/'

	# load clinical
	fname = 'nationwidechildrens.org_clinical_patient_prad.txt'
	fname = datadir+fname
	clinical = pd.read_csv(fname, sep='\t', header=None, skiprows=3)
	clinical_main_header = pd.read_csv(fname, sep='\t', header=None, nrows=1).values[0]
	clinical.columns = clinical_main_header
	print clinical.shape

	# save some columns and create some new
	clinical_small = clinical[['bcr_patient_barcode','gleason_pattern_primary','gleason_pattern_secondary', 'gleason_score']].copy()
	# create grade group column
	clinical_small['grade_group'] = -9999
	clinical_small.loc[clinical_small['gleason_score'] <= 6,'grade_group'] = 1
	clinical_small.loc[(clinical_small['gleason_pattern_primary'] == 3 ) & (clinical_small['gleason_pattern_secondary'] == 4), 'grade_group'] = 2
	clinical_small.loc[(clinical_small['gleason_pattern_primary'] == 4 ) & (clinical_small['gleason_pattern_secondary'] == 3), 'grade_group'] = 3
	clinical_small.loc[clinical_small['gleason_score'] == 8,'grade_group'] = 4
	clinical_small.loc[clinical_small['gleason_score'] >= 9,'grade_group'] = 5
	print ' - Invalid gleason group values: ' + repr(any(clinical_small.grade_group == -9999 ))

	# sort table with grade group
	clinical_small.sort_values(['grade_group'], ascending=[0], inplace=True)
	clinical_small.reset_index(drop=True, inplace= True)

	# create a column with the correct grade group order
	clinical_small['grgr_order'] = range(clinical_small.shape[0])

	# create dictionary of patient barcodes and their order
	patient_order = dict( (clinical_small.bcr_patient_barcode[i],clinical_small.grgr_order[i]) for i in range(clinical_small.shape[0]) )

	# save pandas(csv) and dict (json)
	fname = datadir_out+'clinical'
	clinical_small.to_csv(fname+'.txt', sep='\t')
	print 'saved: '+fname+'.txt'

	with open(fname+'.json', 'w') as fp:
		json.dump(patient_order, fp, indent=4)
	print 'saved: '+fname+'.json'

	return clinical_small, patient_order


# clinical_table, patients_dict = get_clinical()
# clinical_table.shape
# clinical_table.head(1)
