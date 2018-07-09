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
def compute_euclidean_distances(data, data_name, datadir):

    mydata = data.values.T.copy()

    orphanrows = np.where(abs(mydata).sum(axis=1) == 0)[0]
    if (len(orphanrows) > 0):
        logger.info('WARNING: cannot calculate correlation with zero vectors')
        num_string = str(np.sort(np.unique(abs(mydata.flatten())))[1])
        if '.' in num_string:
            dec = len(num_string.rsplit('.')[1].rsplit('0'))
        min_val = float('{:.{prec}f}'.format(0, prec=dec+1)+'1')
        logger.info('  -> replacing zero vectors with value: '+str(min_val))
        mydata[orphanrows, :] = mydata[orphanrows, :] + min_val

    logger.info(' -computing genes euclidean distances...')
    start_time = timeit.default_timer()
    euclidean_distances_condensed = pdist(mydata, 'euclidean')
    logger.info('  >time: '+str(timeit.default_timer() - start_time))

    # save so you don't have to compute again next time
    fpath = datadir+data_name+'__genes_pdistEucl.h5'
    with h5py.File(fpath, 'w') as hf:
        hf.create_dataset('euclidean_distances_condensed',
                          data=euclidean_distances_condensed)
    logger.info(' -saved genes euclidean distances: '+fpath)

    return euclidean_distances_condensed


def load_euclidean_distances(data_name, datadir):

    fpath = os.path.join(datadir, data_name+'__genes_pdistEucl.h5')

    if os.path.exists(fpath):
        # exists
        with h5py.File(fpath, 'r') as hf:
            logger.debug('List of arrays in this file: \n'+str(hf.keys()))
            dataset = hf.get('euclidean_distances_condensed')
            euclidean_distances_condensed = np.array(dataset)
            logger.debug('Shape of the condensed array: ' +
                         str(len(euclidean_distances_condensed)))

        logger.info(' -loaded genes euclidean distances: '+fpath)
        return euclidean_distances_condensed
    else:
        # doesn't exist
        logger.error('euclidean_distances have not been computed!')
        raise


def get_duplicate_cols(data, euclidean_distances_condensed):
    logger.info(' -finding duplicate genes...')
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


def remove_duplicate_cols(data, dupldict, uniqSet, allduplSet):
    setdiff = allduplSet - set(dupldict.keys())
    newdata = data.drop(data[list(setdiff)], axis=1)

    logger.info(' -data size w/ duplicates: '+str(data.shape))
    logger.info(' -genes  w/o duplicates (unique): '+str(len(uniqSet)))
    logger.info(' -all duplicates: '+str(len(allduplSet)))
    logger.info(' -duplicated genes kept: '+str(len(dupldict.keys())))
    logger.info(' -duplicated genes discarded: '+str(len(setdiff)))
    logger.info(' -data size w/o duplicates: '+str(newdata.shape))

    return newdata


def remove_andSave_duplicates(indata, indir, fromFile=True,
                              toCompute_euclidean_distances=False,
                              toSave=False, toPrint=False):
    # load Mutations
    if fromFile:
        inName = indata
        outdir = indir[:]
        fpath = indir+inName

        logger.info('Load data from file: '+fpath)
        data = pd.read_csv(fpath+'.txt', delimiter='\t', index_col=0)
    else:
        inName = indir.rsplit('/')[-1]
        outdir = indir.rstrip(inName)

        logger.info('Load data')
        data = indata.copy()

    logger.info(' -size with duplicates: '+str(data.shape))

    if ('mut' in inName):
        isMut = True
    else:
        isMut = False

    # imgpath = outdir+'img/'
    # if toSave:
    #     imgpath = set_directory(imgpath)
    # if toPlot:
    #     genes_positions_table = load_gene_positions(datadir = outdir)
    #     if data.min().min() < 0:
    #         cmap = 'div'
    #     else:
    #         cmap = 'pos'
    #     title='data before checking for duplicates - '
    #     plot_dist_and_heatmap(data, genes_positions_table, title=title, isMut=isMut, cmap=cmap, saveimg = toSave, showimg = toPrint, imgpath = imgpath+inName+'_wDupl')
    #     plt.close('all')

    if toCompute_euclidean_distances:
        euclidean_distances_condensed = \
            compute_euclidean_distances(data, data_name=inName, datadir=outdir)
        # euclidean_distances = squareform(euclidean_distances_condensed )
    else:
        euclidean_distances_condensed = \
            load_euclidean_distances(inName, datadir=outdir)
        # euclidean_distances = squareform(euclidean_distances_condensed)

    dupldict, uniqSet, allduplSet = \
        get_duplicate_cols(data, euclidean_distances_condensed)
    newdata = remove_duplicate_cols(data, dupldict, uniqSet, allduplSet)

    # if toPlot:
    #     if newdata.min().min() < 0:
    #         cmap = 'div'
    #     else:
    #         cmap = 'pos'
    #     title='data UNIQ columns - '
    #     plot_dist_and_heatmap(newdata, genes_positions_table, title=title, isMut=isMut, cmap=cmap, saveimg = toSave, showimg = toPrint, imgpath = imgpath+inName+'_uniq')
    #     plt.close('all')

    if toSave:
        # saving data
        fpath = outdir + inName + '__uniq'
        logger.info('data and dictionary output will be saved in: '+fpath)

        newdata.to_csv(fpath+'.txt', sep='\t')  # data
        logger.info(' -saved: '+fpath+'.txt')

        with open(fpath+'.json', 'w') as f:
            json.dump(dupldict, f, indent=4)   # dictionary
        logger.info(' -saved: '+fpath+'.json')

    return newdata, dupldict, uniqSet, allduplSet
