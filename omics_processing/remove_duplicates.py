import os
import numpy as np
import pandas as pd
import timeit
import h5py
import json
from scipy.spatial.distance import pdist, squareform
import logging
from .utils import stoc_v, ctos_v
from .io import set_directory

script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


# compute euclidean pairwise distances between genes
# (in order to remove duplicates later)
def _compute_euclidean_distances(data, euclidean_distances_fpath,
                                 to_save_euclidean_distances):

    mydata = data.values.T.copy()

    orphanrows = np.where(abs(mydata).sum(axis=1) == 0)[0]
    if (len(orphanrows) > 0):
        logger.warning('cannot calculate correlation with zero columns!')
        num_string = str(np.sort(np.unique(abs(mydata.flatten())))[1])
        if '.' in num_string:
            dec = len(num_string.rsplit('.')[1].rsplit('0'))
        min_val = float('{:.{prec}f}'.format(0, prec=dec+1)+'1')
        logger.info('replace zero vectors with value: '+str(min_val))
        mydata[orphanrows, :] = mydata[orphanrows, :] + min_val

    logger.info('computing genes euclidean distances...')
    start_time = timeit.default_timer()
    euclidean_distances_condensed = pdist(mydata, 'euclidean')
    logger.info('>time: '+str(timeit.default_timer() - start_time))

    if to_save_euclidean_distances:
        # save so you don't have to compute again next time
        with h5py.File(euclidean_distances_fpath, 'w') as hf:
            hf.create_dataset('euclidean_distances_condensed',
                              data=euclidean_distances_condensed)
        logger.info('save genes euclidean distances in: ' +
                    euclidean_distances_fpath)

    return euclidean_distances_condensed


def _load_euclidean_distances(data_name, euclidean_distances_fpath):

    if os.path.exists(fpath):
        # exists
        with h5py.File(euclidean_distances_fpath, 'r') as hf:
            logger.debug('List of arrays in this file: \n'+str(hf.keys()))
            dataset = hf.get('euclidean_distances_condensed')
            euclidean_distances_condensed = np.array(dataset)
            logger.debug('Shape of the condensed array: ' +
                         str(len(euclidean_distances_condensed)))

        logger.info('loaded genes euclidean distances from: ' +
                    euclidean_distances_fpath)
        return euclidean_distances_condensed
    else:
        # doesn't exist
        logger.error('euclidean_distances have not been computed!')
        raise


def _get_duplicate_cols(data, euclidean_distances_condensed):
    logger.info('finding duplicate genes...')
    # for every gene
    dupldict = {}
    uniqSet = set([])
    allduplSet = set([])
    n = data.shape[1]
    for genename in data.columns.values:
        # calculate the euclidean distance
        # between that gene and every other gene
        geneloc = data.columns.get_loc(genename)
        cidx_gene = stoc_v(geneloc, n)
        cidx_zero = cidx_gene[
            np.where(euclidean_distances_condensed[cidx_gene] == 0)[0]]
        idx = ctos_v(geneloc, cidx_zero, n)
        if len(idx) == 0:
            uniqSet.add(genename)
        else:
            if genename not in allduplSet:
                allduplSet.add(genename)
                # get the genes that are identical (distance == 0)
                duplgenes = data.iloc[:, idx].columns.values.tolist()
                allduplSet.update(duplgenes)
                # keep record of this gene's duplicates in a dictionary
                dupldict[genename] = duplgenes
    return dupldict, uniqSet, allduplSet


def _remove_duplicate_cols(data, dupldict, uniqSet, allduplSet):
    logger.info('removing duplicate genes...')
    setdiff = allduplSet - set(dupldict.keys())
    newdata = data.drop(data[list(setdiff)], axis=1)

    logger.info(' -data shape w/ duplicates: '+str(data.shape))
    logger.info(' -number of genes w/o duplicates (unique): ' +
                str(len(uniqSet)))
    logger.info(' -all gene duplicates: '+str(len(allduplSet)))
    logger.info(' -duplicated genes kept (the first by chr order): ' +
                str(len(dupldict.keys())))
    logger.info(' -duplicated genes discarded (the rest): '+str(len(setdiff)))
    logger.info(' -data shape w/o duplicates: '+str(newdata.shape))

    return newdata


def remove_andSave_duplicates(indata, **kwargs):
    load_from_file = kwargs.get('load_from_file', False)
    to_compute_euclidean_distances = \
        kwargs.get('to_compute_euclidean_distances', True)
    to_save_euclidean_distances = \
        kwargs.get('to_save_euclidean_distances', False)
    to_save_output = kwargs.get('to_save_output', True)

    output_filename = kwargs.get('output_filename', 'data_wo_duplicates')
    output_directory = kwargs.get('output_directory',
                                  os.path.join(script_path, "..", "data",
                                               "processed"))

    euclidean_distances_fpath = os.path.join(output_directory,
                                             output_filename +
                                             '__genes_pdist_eucl.h5')
    # load data
    if load_from_file:
        fpath = indata
        logger.info('Load data from file: '+fpath)
        data = pd.read_csv(fpath+'.txt', delimiter='\t', index_col=0)
    else:
        logger.info('Load data from input.')
        data = indata.copy()

    logger.info('size before checking for duplicate columns: '+str(data.shape))

    if to_compute_euclidean_distances:
        euclidean_distances_condensed = \
            _compute_euclidean_distances(data, euclidean_distances_fpath,
                                         to_save_euclidean_distances)
        # euclidean_distances = squareform(euclidean_distances_condensed )
    else:
        euclidean_distances_condensed = \
            _load_euclidean_distances(euclidean_distances_fpath)
        # euclidean_distances = squareform(euclidean_distances_condensed)

    dupldict, uniqSet, allduplSet = \
        _get_duplicate_cols(data, euclidean_distances_condensed)
    newdata = _remove_duplicate_cols(data, dupldict, uniqSet, allduplSet)

    if to_save_output:
        # saving data
        fpath = os.path.join(output_directory,
                             output_filename +
                             '__uniq')
        logger.info('saving data and dictionary output...')

        newdata.to_csv(fpath+'.txt', sep='\t')  # data
        logger.info('saved: '+fpath+'.txt')

        with open(fpath+'.json', 'w') as f:
            json.dump(dupldict, f, indent=4)   # dictionary
        logger.info('saved: '+fpath+'.json')

    return newdata, dupldict, uniqSet, allduplSet
