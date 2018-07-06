import os
import numpy as np
import pandas as pd
import timeit
import h5py
import json
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting_utils import plot_dist_and_heatmap
from utils.load_functions import load_gene_positions
from utils.math_utils import stoc_v, ctos_v

#### DEPRECATED: remove duplicates without saving them
# def remove_duplicates(data_name, toPlot=False,
#                       # define dirs
#                       datadir = 'DATA/TCGA_fromMatteo/all_processed/formatted/'+subDir,
#                       datadir_out = 'DATA/TCGA_fromMatteo/all_processed/formatted/'+subDir+'uniq/'
#                      ):
#     # load Mutations
#     fname = data_name+'.txt'
#     fname = datadir+fname
#     data = pd.read_csv(fname, delimiter='\t', index_col=0)
#     print('size with duplicates: '+str(data.shape)
#     if toPlot:
#         if ('mut' in data_name):
#             plot_dist_and_heatmap(data,  isMut=True)
#         else:
#             plot_dist_and_heatmap(data, isMut=False)
#     data = data.T.drop_duplicates().T
#     print('size without duplicates: '+str(data.shape)
#     if toPlot:
#         title='UNIQ: '
#         if ('mut' in data_name):
#             plot_dist_and_heatmap(data, title=title, isMut=True)
#         else:
#             plot_dist_and_heatmap(data, title=title, isMut=False)
#     # saving data
#     fname = datadir_out+data_name+'.txt'
#     data.to_csv(fname, sep='\t')
#     print('saved: '+fname
#     return data

#compute euclidean pairwise distances between genes (in order to remove duplicates later)
def compute_euclidean_distances(data, data_name,
								# define dirs
								datadir = 'DATA/TCGA_fromMatteo/all_processed/formatted/uniq/'
								):

	mydata = data.values.T.copy()

	orphanrows = np.where(abs(mydata).sum(axis = 1)==0)[0]
	if (len(orphanrows) > 0):
		print('WARNING: cannot calculate correlation with zero vectors')
		num_string = str(np.sort(np.unique(abs(mydata.flatten())))[1])
		if '.' in num_string:
			dec = len(num_string.rsplit('.')[1].rsplit('0'))
		min_val = float('{:.{prec}f}'.format(0, prec=dec+1)+'1')
		print('  -> replacing zero vectors with value: '+str(min_val))
		mydata[orphanrows,:] = mydata[orphanrows,:] + min_val

	print(' -computing genes euclidean distances...')
	start_time = timeit.default_timer()
	euclidean_distances_condensed  = pdist(mydata, 'euclidean')
	print('  >time: '+str(timeit.default_timer() - start_time))

	# save so you don't have to compute again next time
	fpath = datadir+data_name+'__genes_pdistEucl.h5'
	with h5py.File(fpath, 'w') as hf:
		hf.create_dataset('euclidean_distances_condensed', data=euclidean_distances_condensed)
	print(' -saved genes euclidean distances: '+fpath)

	return euclidean_distances_condensed

def load_euclidean_distances(data_name,
							# define dirs
							datadir = 'DATA/TCGA_fromMatteo/all_processed/formatted/uniq/'
							):


	fpath = datadir+data_name+'__genes_pdistEucl.h5'

	if os.path.exists(fpath):
		# exists
		with h5py.File(fpath,'r') as hf:
			#print('List of arrays in this file: \n'+str(hf.keys())
			dataset = hf.get('euclidean_distances_condensed')
			euclidean_distances_condensed = np.array(dataset)
			#print('Shape of the condensed array: '+str(len(euclidean_distances_condensed))

		print(' -loaded genes euclidean distances: '+fpath)
		return euclidean_distances_condensed
	else:
		# doesn't exist
		print('ERROR: euclidean_distances have not been computed')
		raise


# def get_duplicate_cols(data, euclidean_distances_condensed):
#     print(' -finding duplicate genes...')
#     # for every gene
#     dupldict = {}
#     uniqSet = set([])
#     allgenesSet = set([])
#     for genename in data.columns.values:
#         # calculate the euclidean distance between that gene and every other gene
#         geneloc = data.columns.get_loc(genename)
#         idx = np.where(euclidean_distances[geneloc,:] == 0)[0]
#         if len(idx) > 0:
#             if len(idx) == 1:
#                 uniqSet.add(genename)
#             if not genename in allgenesSet:
#                 allgenesSet.add(genename)
#                 # get the genenames that are identical (distance == 0)
#                 duplgenes = data.iloc[:,idx].columns.values.tolist()
#                 allgenesSet.update(duplgenes)
#                 # save ALL these genes in the duplicates dictionary
#                 dupldict[genename] = duplgenes
#
#
#     return dupldict, uniqSet, allgenesSet

def get_duplicate_cols(data, euclidean_distances_condensed):
	print(' -finding duplicate genes...')
	# for every gene
	dupldict = {}
	uniqSet = set([])
	allduplSet = set([])
	n = data.shape[1]
	for genename in data.columns.values:
		# calculate the euclidean distance between that gene and every other gene
		geneloc = data.columns.get_loc(genename)
		cidx_gene = stoc_v(geneloc,n)
		cidx_zero = cidx_gene[ np.where(euclidean_distances_condensed[cidx_gene] == 0)[0] ]
		idx = ctos_v(geneloc, cidx_zero, n)
		if len(idx) == 0:
			uniqSet.add(genename)
		else:
			if not genename in allduplSet:
				allduplSet.add(genename)
				# get the genes that are identical (distance == 0)
				duplgenes = data.iloc[:,idx].columns.values.tolist()
				allduplSet.update(duplgenes)
				# keep record of this gene's duplicates in the duplicates dictionary
				dupldict[genename] = duplgenes
	return dupldict, uniqSet, allduplSet

def remove_duplicate_cols(data, dupldict, uniqSet, allduplSet):
	setdiff = allduplSet - set(dupldict.keys())
	newdata = data.drop(data[list(setdiff)], axis=1)

	print(' -data size w/ duplicates: '+str( data.shape ))
	print(' -genes  w/o duplicates (unique): '+str( len(uniqSet) ))
	print(' -all duplicates: '+str( len(allduplSet) ))
	print(' -duplicated genes kept: '+str( len(dupldict.keys()) ))
	print(' -duplicated genes discarded: '+str( len(setdiff) ))
	print(' -data size w/o duplicates: '+str( newdata.shape ))

	return newdata


# def remove_andSave_duplicates(data_name, toCompute_euclidean_distances=False, toPlot=False, toSave = False,
#                               # define dirs
#                               datadir = 'DATA/TCGA_fromMatteo/all_processed/formatted/',
#                               subdir = 'split/',
#                               saveimg = False, imgpath = None
#                              ):

def remove_andSave_duplicates(indata, fromFile = True,
							  toCompute_euclidean_distances=False, toPlot=False, toSave = False, toPrint = False,
							  indir = 'DATA/TCGA_fromMatteo/all_processed/formatted/'):
	# load Mutations
	if fromFile:
		inName = indata
		outdir = indir[:]
		fpath = indir+inName

		print('Load data from file: '+fpath)
		data = pd.read_csv(fpath+'.txt', delimiter='\t', index_col=0)
	else:
		inName = indir.rsplit('/')[-1]
		outdir = indir.rstrip(inName)

		print('Load data')
		data = indata.copy()

	print(' -size with duplicates: '+str(data.shape))

	if ('mut' in inName):
		isMut = True
	else:
		isMut = False

	imgpath = outdir+'img/'
	if toSave:
	  if not os.path.exists(imgpath):
		  os.makedirs(imgpath)
		  print(' >created '+imgpath)


	if toPlot:
		genes_positions_table = load_gene_positions(datadir = outdir)
		if data.min().min() < 0:
			cmap = 'div'
		else:
			cmap = 'pos'
		title='data before checking for duplicates - '
		plot_dist_and_heatmap(data, genes_positions_table, title=title, isMut=isMut, cmap=cmap, saveimg = toSave, showimg = toPrint, imgpath = imgpath+inName+'_wDupl')
		plt.close('all')

	if toCompute_euclidean_distances:
		euclidean_distances_condensed = compute_euclidean_distances(data, data_name = inName, datadir=outdir)
		# euclidean_distances = squareform(euclidean_distances_condensed )
	else:
		euclidean_distances_condensed = load_euclidean_distances(inName, datadir=outdir)
		# euclidean_distances = squareform(euclidean_distances_condensed)

	dupldict, uniqSet, allduplSet = get_duplicate_cols(data, euclidean_distances_condensed)
	newdata = remove_duplicate_cols(data, dupldict, uniqSet, allduplSet)

	if toPlot:
		if newdata.min().min() < 0:
			cmap = 'div'
		else:
			cmap = 'pos'
		title='data UNIQ columns - '
		plot_dist_and_heatmap(newdata, genes_positions_table, title=title, isMut=isMut, cmap=cmap, saveimg = toSave, showimg = toPrint, imgpath = imgpath+inName+'_uniq')
		plt.close('all')

	if toSave:
		# saving data
		fpath = outdir+inName+ '__uniq'
		print('data and dictionary output will be saved in: '+fpath)

		newdata.to_csv(fpath+'.txt', sep='\t') ## data
		print(' -saved: '+fpath+'.txt')

		with open(fpath+'.json', 'w') as f:
			# test = json.dumps(dupldict)
			# print(test)
			# f.write(test)
			json.dump(dupldict, f, indent=4)  ## dictionary
		print(' -saved: '+fpath+'.json')

	return newdata, dupldict, uniqSet, allduplSet
