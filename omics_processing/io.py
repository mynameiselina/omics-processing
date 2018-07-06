import os
import numpy as np
import pandas as pd
import json
from natsort import natsorted, index_natsorted


# if folder does not exist, create it
def set_directory(mydir):
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        print(' >created folder: '+mydir)
    return mydir


# RUN JUST ONE TIME
def get_clinical(fpath, datadir_out):

    # define dirs
    # maindir = '/home/ubuntu/my-git-repos/netDL/'
    # datadir = maindir+'DATA/TCGA_fromMatteo/all_processed/'
    # datadir_out = maindir+'DATA/TCGA_fromMatteo/all_processed/formatted/'

    # load clinical
    # fname = 'nationwidechildrens.org_clinical_patient_prad.txt'
    # fname = datadir+fname
    clinical = pd.read_csv(fpath, sep='\t', header=None, skiprows=3)
    clinical_main_header = pd.read_csv(fpath, sep='\t', header=None,
                                       nrows=1).values[0]
    clinical.columns = clinical_main_header

    # save some columns and create some new
    clinical_small = clinical[['bcr_patient_barcode','gleason_pattern_primary',
                               'gleason_pattern_secondary', 'gleason_score']
                              ].copy()
    # create grade group column
    clinical_small['grade_group'] = -9999
    clinical_small.loc[clinical_small['gleason_score'] <= 6, 'grade_group'] = 1
    clinical_small.loc[(clinical_small['gleason_pattern_primary'] == 3) &
                       (clinical_small['gleason_pattern_secondary'] == 4),
                       'grade_group'] = 2
    clinical_small.loc[(clinical_small['gleason_pattern_primary'] == 4) &
                       (clinical_small['gleason_pattern_secondary'] == 3),
                       'grade_group'] = 3
    clinical_small.loc[clinical_small['gleason_score'] == 8, 'grade_group'] = 4
    clinical_small.loc[clinical_small['gleason_score'] >= 9, 'grade_group'] = 5
    print(' - Invalid gleason group values: ' +
          repr(any(clinical_small.grade_group == -9999)))

    # sort table with grade group
    clinical_small.sort_values(['grade_group'], ascending=[0], inplace=True)
    clinical_small.reset_index(drop=True, inplace=True)

    # create a column with the correct grade group order
    clinical_small['grgr_order'] = range(clinical_small.shape[0])

    # create dictionary of patient barcodes and their order
    patient_order = dict((clinical_small.bcr_patient_barcode[i],
                          clinical_small.grgr_order[i])
                         for i in range(clinical_small.shape[0]))

    # save pandas(csv) and dict (json)
    datadir_out = set_directory(datadir_out)
    fpath_out = datadir_out+'clinical'
    clinical_small.to_csv(fpath_out+'.txt', sep='\t')
    print 'saved: '+fpath_out+'.txt'

    with open(fpath_out+'.json', 'w') as fp:
        json.dump(patient_order, fp, indent=4)
    print 'saved: '+fpath_out+'.json'

    return clinical_small, patient_order


# RUN JUST ONE TIME
def get_gaf(fpath, datadir_out):

    # define dirs
    # maindir = '/home/ubuntu/my-git-repos/netDL/'
    # datadir = maindir+'DATA/TCGA_fromMatteo/all_processed/'
    # datadir_out = maindir+'DATA/TCGA_fromMatteo/all_processed/formatted/'

    # load GAF
    # fname = 'gene.genome.v4_0.processed.gaf'
    gaf = pd.read_csv(fpath, delimiter='\t', header=None)

    # save some columns of GAF and create new ones
    gaf_small = gaf.iloc[:, [1, 16]].copy()

    gaf_small['gene'] = [item.rsplit('|')[0] 
                         for item in gaf_small.iloc[:, 0]]
    gaf_small['ensemble'] = [item.rsplit('|')[1]
                             for item in gaf_small.iloc[:, 0]]
    gaf_small['chr'] = [item.rsplit(':')[0].rsplit('chr')[1] 
                        for item in gaf_small[16]]
    gaf_small['start'] = [int(item.rsplit(':')[1].rsplit('-')[0])
                          for item in gaf_small[16]]
    gaf_small['end'] = [item.rsplit(':')[1].rsplit('-')[1]
                        for item in gaf_small[16]]
    gaf_small['sign'] = [item.rsplit(':')[2] for item in gaf_small[16]]
    gaf_small['gene_chr'] = gaf_small['gene']+'_'+gaf_small['chr']

    # sort GAF by gene name - to deal with the duplicates
    gaf_small.sort_values(['gene_chr'], ascending=[1], inplace=True)

    # show me all duplicate gene names
    print('all duplicates: ' +
          str(gaf_small[gaf_small.gene.duplicated(keep=False)].shape))
    print('the gene_chr is always duplicated only once more: ' +
          str((gaf_small[gaf_small.gene.duplicated(keep=False)
                         ].gene_chr.value_counts() == 2).all()))

    # of the duplicates keep only one copy
    # and save the min start pos and the max end pos
    gaf_posMerged = gaf_small.copy()  # need to create a copy and not view here
    init = gaf_posMerged.shape
    print('initial gaf size '+str(init))
    print(' -duplicates exist : '+str(not(gaf_small.gene.value_counts() == 1
                                          ).all()))
    # VIEW of all duplicates
    gaf_dupl = gaf_posMerged[gaf_posMerged['gene_chr'].duplicated()]
    tmp = gaf_posMerged[gaf_posMerged['gene_chr'].duplicated()].shape
    print ' -number of all duplicates '+str(tmp)
    # keep only the first duplicate and drop the rest
    gaf_posMerged.drop_duplicates(['gene_chr'], inplace=True)
    print('gaf size without duplicates '+str(gaf_posMerged.shape))
    print(' -successfully removed duplicates: ' +
          str((gaf_posMerged.gene.value_counts() == 1).all()))
    print(' -numbers match: ' +
          str((init[0] - gaf_posMerged.shape[0]) == tmp[0]))
    gaf_posMerged.reset_index(drop=True, inplace=True)

    # save min start and max end from all duplicated gene_chr entries
    _dupl_genes = gaf_small[gaf_small.gene.duplicated(keep=False)]
    gaf_dupl.start = _dupl_genes.groupby(['gene_chr'], sort=False
                                         )['start'].min().values
    gaf_dupl.end = _dupl_genes.groupby(['gene_chr'], sort=False
                                       )['end'].max().values
    print(' -updated with the merged start and end positions')

    # sort GAF by pos (chr and start-end)

    gaf_posMerged.sort_values(['start', 'end'], ascending=[1, 1], inplace=True)
    chr_order = index_natsorted(gaf_posMerged.chr.values)
    gaf_posMerged = gaf_posMerged.iloc[chr_order, :]
    gaf_posMerged.reset_index(drop=True, inplace=True)

    # create a column with the correct position order
    gaf_posMerged['pos_order'] = range(gaf_posMerged.shape[0])

    # create dictionary of gene names and their order
    gene_order = dict((gaf_posMerged.gene[i], gaf_posMerged.pos_order[i])
                      for i in range(gaf_posMerged.shape[0]))

    # create dictionary of gene names and their chr
    gene_chr = dict((gaf_posMerged.gene[i], gaf_posMerged.chr[i])
                    for i in range(gaf_posMerged.shape[0]))

    # save pandas(csv) and dict(json)
    datadir_out = set_directory(datadir_out)
    fpath_out = datadir_out+'gaf'
    gaf_posMerged.to_csv(fpath_out+'.txt', sep='\t')
    print('saved: '+fpath_out+'.txt')

    with open(fpath_out+'.json', 'w') as fp:
        json.dump(gene_order, fp, indent=4)
    print('saved: '+fpath_out+'.json')

    with open(fpath_out+'_chr.json', 'w') as fp:
        json.dump(gene_chr, fp, indent=4)
    print('saved: '+fpath_out+'_chr.json')

    return gaf_posMerged, gene_order, gene_chr
