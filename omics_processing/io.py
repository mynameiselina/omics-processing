import os
import numpy as np
import pandas as pd
import json
from natsort import natsorted, index_natsorted
import logging

logger = logging.getLogger(__name__)


# if folder does not exist, create it
def set_directory(mydir):
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        logger.info(' >created folder: '+mydir)
    return mydir


# RUN JUST ONE TIME
def process_clinical_TCGA(
        fpath, datadir_out, key_col, **kwargs):

    old_file_format = kwargs.pop('old_file_format', False)
    if old_file_format:
        clinical = pd.read_csv(fpath, sep='\t', header=None, skiprows=3)
        clinical.columns = pd.read_csv(fpath, sep='\t', header=None,
                                       nrows=1).values[0]
    else:
        clinical = pd.read_csv(fpath, sep='\t', header=None, index_col=0).T
        clinical.set_index(key_col, inplace=True)

        # clinical[key_col+'_upper'] = clinical.index.str.upper()
        # clinical.set_index(key_col+'_upper', drop =True, inplace = True)

    select_columns = []

    gleason_cols = kwargs.pop('gleason_cols', None)
    if gleason_cols is not None:
        if (('score' not in gleason_cols) or
                ('primary' not in gleason_cols) or
                ('secondary' not in gleason_cols)):
            logger.error(
                'Invalid gleason columns to compute the gleason score',
                gleason_cols)
            raise
        split_gleason_cols = gleason_cols.rsplit(',')
        select_columns.extend(split_gleason_cols)
        logger.debug("Keep columns from clinical table:\n"+str(select_columns))

    other_cols = kwargs.pop('other_cols', None)
    if other_cols is not None:
        split_other_cols = other_cols.rsplit(',')
        keep_cols = [c for c in split_other_cols
                     if ((c not in select_columns) & (c != ''))]
        select_columns.extend(keep_cols)
        logger.debug(select_columns)

    # save some columns and create some new
    clinical_small = clinical[select_columns].copy()

    if gleason_cols is not None:
        for col in split_gleason_cols:
            if 'score' in col:
                gleason_score_col = col
            elif 'primary' in col:
                gleason_primary_col = col
            elif 'secondary' in col:
                gleason_secondary_col = col

        # extract score values
        score = clinical_small[gleason_score_col].astype(float)
        primary = clinical_small[gleason_primary_col].astype(float)
        secondary = clinical_small[gleason_secondary_col].astype(float)

        # create grade group column
        clinical_small['grade_group'] = -9999
        clinical_small.loc[score <= 6, 'grade_group'] = 1
        clinical_small.loc[(primary == 3) & (secondary == 4),
                           'grade_group'] = 2
        clinical_small.loc[(primary == 4) & (secondary == 3),
                           'grade_group'] = 3
        clinical_small.loc[score == 8, 'grade_group'] = 4
        clinical_small.loc[score >= 9, 'grade_group'] = 5
        _invalid_values = (clinical_small.grade_group == -9999)
        if any(_invalid_values):
            logger.info(
                'Invalid gleason group values for samples:\n' +
                clinical_small.index[_invalid_values])

    # save pandas(csv) and dict (json)
    datadir_out = set_directory(datadir_out)
    fpath_out = os.path.join(datadir_out, 'clinical.txt')
    clinical_small.to_csv(fpath_out, sep='\t')
    logger.info('saved: '+fpath_out)

    return clinical_small


# RUN JUST ONE TIME
def process_TCGA_gaf(fpath, datadir_out, **read_csv_kwargs):

    # load GAF
    gaf = pd.read_csv(fpath, **read_csv_kwargs)

    # save some columns of GAF and create new ones
    gaf_small = gaf.iloc[:, [1, 16]].copy()

    gaf_small.loc[:, 'gene'] = [item.rsplit('|')[0]
                                for item in gaf_small.iloc[:, 0]]
    gaf_small.loc[:, 'ensemble'] = [item.rsplit('|')[1]
                                    for item in gaf_small.iloc[:, 0]]
    gaf_small.loc[:, 'chr'] = [item.rsplit(':')[0].rsplit('chr')[1]
                               for item in gaf_small.iloc[:, 1]]
    gaf_small.loc[:, 'start'] = [int(item.rsplit(':')[1].rsplit('-')[0])
                                 for item in gaf_small.iloc[:, 1]]
    gaf_small.loc[:, 'end'] = [item.rsplit(':')[1].rsplit('-')[1]
                               for item in gaf_small.iloc[:, 1]]
    gaf_small.loc[:, 'sign'] = [item.rsplit(':')[2]
                                for item in gaf_small.iloc[:, 1]]
    gaf_small.loc[:, 'gene_chr'] = \
        gaf_small.loc[:, 'gene']+'_'+gaf_small.loc[:, 'chr']

    # sort GAF by gene name - to deal with the duplicates
    gaf_small.sort_values(['gene_chr'], ascending=[1], inplace=True)

    # show me all duplicate gene names
    logger.info('all duplicates: ' +
                str(gaf_small[gaf_small.gene.duplicated(keep=False)].shape))
    logger.info('the gene_chr is always duplicated only once more: ' +
                str((gaf_small[gaf_small.gene.duplicated(keep=False)
                               ].gene_chr.value_counts() == 2).all()))

    # of the duplicates keep only one copy
    # and save the min start pos and the max end pos
    gaf_posMerged = gaf_small.copy()  # need to create a copy and not view here
    size = gaf_posMerged.shape
    logger.info('initial gaf size '+str(size))
    logger.info(' -duplicates exist : ' +
                str(not(gaf_small.gene.value_counts() == 1).all()))
    # VIEW of all duplicates
    gaf_dupl = gaf_posMerged[gaf_posMerged['gene_chr'].duplicated()].copy()
    tmp_size = gaf_dupl.shape
    logger.info(' -number of all duplicates '+str(tmp_size))

    # keep only the first duplicate and drop the rest
    gaf_posMerged.drop_duplicates(['gene_chr'], inplace=True)
    logger.info('gaf size without duplicates '+str(gaf_posMerged.shape))
    logger.info(' -successfully removed duplicates: ' +
                str((gaf_posMerged.gene.value_counts() == 1).all()))
    logger.info(' -numbers match: ' +
                str((size[0] - gaf_posMerged.shape[0]) == tmp_size[0]))
    gaf_posMerged.reset_index(drop=True, inplace=True)

    # save min start and max end from all duplicated gene_chr entries
    _dupl_genes = gaf_small[gaf_small.gene.duplicated(keep=False)]
    gaf_dupl.loc[:, 'start'] = _dupl_genes.groupby(['gene_chr'], sort=False
                                                   )['start'].min().values
    gaf_dupl.loc[:, 'end'] = _dupl_genes.groupby(['gene_chr'], sort=False
                                                 )['end'].max().values
    logger.info(' -updated with the merged start and end positions')

    # sort GAF by pos (chr and start-end)

    gaf_posMerged.sort_values(['start', 'end'], ascending=[1, 1], inplace=True)
    chr_order = index_natsorted(gaf_posMerged.chr.values)
    gaf_posMerged = gaf_posMerged.iloc[chr_order, :].copy()
    gaf_posMerged.reset_index(drop=True, inplace=True)

    # create a column with the correct position order
    gaf_posMerged.loc[:, 'pos_order'] = range(gaf_posMerged.shape[0])

    # create dictionary of gene names and their order
    gene_order = dict((gaf_posMerged.gene[i], int(gaf_posMerged.pos_order[i]))
                      for i in range(gaf_posMerged.shape[0]))

    # create dictionary of gene names and their chr
    gene_chr = dict((gaf_posMerged.gene[i], gaf_posMerged.chr[i])
                    for i in range(gaf_posMerged.shape[0]))

    # save pandas(csv) and dict(json)
    datadir_out = set_directory(datadir_out)
    fpath_out = os.path.join(datadir_out, 'gaf')
    gaf_posMerged.to_csv(fpath_out+'.txt', sep='\t')
    logger.info('saved: '+fpath_out+'.txt')

    with open(fpath_out+'.json', 'w') as fp:
        json.dump(gene_order, fp, indent=4)
    logger.info('saved: '+fpath_out+'.json')

    with open(fpath_out+'_chr.json', 'w') as fp:
        json.dump(gene_chr, fp, indent=4)
    logger.info('saved: '+fpath_out+'_chr.json')

    return gaf_posMerged, gene_order, gene_chr


# load the processed gaf file with the genes positions
def load_gene_positions(fpath, **read_csv_kwargs):
    # **{ 'sep': '\t', 'index_col': 0, 'header': 0}
    # gaf file with gene positions
    if not os.path.exists(fpath):
        return None
    else:
        logger.info('Load gaf file: '+fpath)
        genes_positions_table = pd.read_csv(fpath, **read_csv_kwargs)
        genes_positions_table = genes_positions_table[['gene', 'chr']]
        genes_positions_table.columns = ['id', 'chr']

        genes_positions_table['chr_int'] = genes_positions_table['chr']
        di = {"X": 23, "Y": 24, "M": 25}
        genes_positions_table = genes_positions_table.replace({"chr_int": di})
        genes_positions_table['chr_int'] = \
            genes_positions_table['chr_int'].astype(int)
        genes_positions_table['id'] = genes_positions_table['id'].astype(str)

        return genes_positions_table


# load the gene order position
def load_gene_order_dict(fpath):
    # load gene dict
    logger.info('Load gene dict file: '+fpath)
    with open(fpath, 'r') as fp:
        gene_dict = json.load(fp)

    return gene_dict


# load the clinical data from the patients cohort
def load_clinical(fpath, **read_csv_kwargs):
    col_as_index = read_csv_kwargs.pop('col_as_index', None)
    # load clinical to get patient labels on the grade group
    logger.info('Load clinical file: '+fpath)
    clinical = pd.read_csv(fpath, **read_csv_kwargs)
    if col_as_index is not None:
        if col_as_index != clinical.index.name:
            clinical.dropna(subset=[col_as_index], inplace=True)
            clinical.set_index([col_as_index], inplace=True, drop=True)

    return clinical


def set_path(f, parent_dir=None, force=False):
    sep = os.sep
    # if starts on root, then absolute path
    # else, relative path
    start_on_root = False
    if f.startswith(sep):
        start_on_root = True

    # split with '/' and join again (os.path.join)
    f = os.path.join(*f.rsplit(sep))
    if start_on_root:
        f = os.path.join(sep, f)
        f = os.path.abspath(f)
    else:
        if parent_dir is not None:
            parent_dir = os.path.join(*parent_dir.rsplit(sep))
            f = os.path.join(parent_dir, f)

    # if fpath does not exist
    if not os.path.exists(f):
        # if user wants to force it
        if force:
            # create it
            logger.warning(
                f+" fpath does not exist but it will be created!")
            f = set_directory(f)
        else:
            # else give error
            logger.error(
                f+" fpath does not exist!")
            f = None

    return f


def parse_arg_type(arg, type):
    if arg is not None:
        if not isinstance(arg, type):
            if type == bool:
                arg = bool(strtobool(arg))
            else:
                arg = type(arg)
    return arg
