import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')


## RUN JUST ONE TIME
def get_gaf():

	# define dirs
	maindir = '/home/ubuntu/my-git-repos/netDL/'
	datadir = maindir+'DATA/TCGA_fromMatteo/all_processed/'
	datadir_out = maindir+'DATA/TCGA_fromMatteo/all_processed/formatted/'

	# load GAF
	fname = 'gene.genome.v4_0.processed.gaf'
	fname = datadir+fname
	gaf = pd.read_csv(fname, delimiter='\t', header=None)
	print gaf.shape

	# save some columns of GAF and create new ones
	gaf_small = gaf.iloc[:,[1,16]].copy()

	gaf_small['gene'] = [item.rsplit('|')[0] for item in gaf_small.iloc[:,0]]
	gaf_small['ensemble'] = [item.rsplit('|')[1] for item in gaf_small.iloc[:,0]]
	gaf_small['chr'] = [item.rsplit(':')[0].rsplit('chr')[1] for item in gaf_small[16]]
	gaf_small['start'] = [int(item.rsplit(':')[1].rsplit('-')[0]) for item in gaf_small[16]]
	gaf_small['end'] = [item.rsplit(':')[1].rsplit('-')[1] for item in gaf_small[16]]
	gaf_small['sign'] = [item.rsplit(':')[2] for item in gaf_small[16]]
	gaf_small['gene_chr'] = gaf_small['gene']+'_'+gaf_small['chr']


	# sort GAF by gene name - to deal with the duplicates
	gaf_small.sort_values(['gene_chr'], ascending=[1], inplace=True)

	# show me all duplicate gene names
	print 'all duplicates: '+str(gaf_small[gaf_small.gene.duplicated(keep=False)].shape)
	print 'the gene_chr is always duplicated only once more: '+str((gaf_small[gaf_small.gene.duplicated(keep=False)].gene_chr.value_counts() == 2).all())
	# gaf_small[gaf_small.gene.duplicated(keep=False)].head(2)

	# of the duplicates keep only one copy and save the min start pos and the max end pos
	gaf_posMerged = gaf_small.copy() # need to create a copy and not view here
	init = gaf_posMerged.shape; print 'initial gaf size '+str(init)
	print ' -duplicates exist : '+str(not(gaf_small.gene.value_counts() == 1).all())
	gaf_dupl = gaf_posMerged[gaf_posMerged['gene_chr'].duplicated()] # VIEW of all duplicates
	tmp = gaf_posMerged[gaf_posMerged['gene_chr'].duplicated()].shape; print ' -number of all duplicates '+str(tmp)
	gaf_posMerged.drop_duplicates(['gene_chr'], inplace=True) # keep only the first duplicate and drop the rest
	print 'gaf size without duplicates '+str(gaf_posMerged.shape)
	print ' -successfully removed duplicates: '+str((gaf_posMerged.gene.value_counts() == 1).all())
	print ' -numbers match: '+str((init[0] - gaf_posMerged.shape[0]) == tmp[0])
	gaf_posMerged.reset_index(drop=True, inplace= True)

	# save min start and max end from all duplicated gene_chr entries
	gaf_dupl.start = gaf_small[gaf_small.gene.duplicated(keep=False)].groupby(['gene_chr'], sort=False)['start'].min().values
	gaf_dupl.end = gaf_small[gaf_small.gene.duplicated(keep=False)].groupby(['gene_chr'], sort=False)['end'].max().values
	print ' -updated with the merged start and end positions'

	# sort GAF by pos (chr and start-end)
	from natsort import natsorted, index_natsorted
	# a =  ['aX','a2', 'a9', 'a1', 'a4', 'a10','aY']
	# index_natsorted(a)

	gaf_posMerged.sort_values(['start','end'], ascending=[1,1], inplace=True)
	chr_order = index_natsorted(gaf_posMerged.chr.values)
	gaf_posMerged = gaf_posMerged.iloc[chr_order,:]
	gaf_posMerged.reset_index(drop=True, inplace= True)


	# create a column with the correct position order
	gaf_posMerged['pos_order'] = range(gaf_posMerged.shape[0])

	# create dictionary of gene names and their order
	gene_order = dict( (gaf_posMerged.gene[i],gaf_posMerged.pos_order[i]) for i in range(gaf_posMerged.shape[0]) )

	# create dictionary of gene names and their chr
	gene_chr = dict( (gaf_posMerged.gene[i],gaf_posMerged.chr[i]) for i in range(gaf_posMerged.shape[0]) )

	# save pandas(csv) and dict (json)
	fname = datadir_out+'gaf'
	gaf_posMerged.to_csv(fname+'.txt', sep='\t')
	print 'saved: '+fname+'.txt'

	with open(fname+'.json', 'w') as fp:
		json.dump(gene_order, fp, indent=4)
	print 'saved: '+fname+'.json'

	with open(fname+'_chr.json', 'w') as fp:
		json.dump(gene_chr, fp, indent=4)
	print 'saved: '+fname+'_chr.json'

	return gaf_posMerged, gene_order, gene_chr


# gaf_table, gene_order, gene_chr = get_gaf()
# gaf_table.shape
# len(gene_dict)
