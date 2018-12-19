import os
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


# if folder does not exist, create it
def set_directory(mydir):
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        logger.debug(' >created folder: '+mydir)
    return mydir


# load the gene order position
def load_gene_order_dict(fpath):
    # load gene dict
    logger.debug('Load gene dict file: '+fpath)
    with open(fpath, 'r') as fp:
        gene_dict = json.load(fp)

    return gene_dict


# load the clinical data from the patients cohort
def load_clinical(fpath, **read_csv_kwargs):
    col_as_index = read_csv_kwargs.pop('col_as_index', None)
    # load clinical to get patient labels on the grade group
    logger.debug('Load clinical file: '+fpath)
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
            parent_dir_bak = parent_dir[:]
            parent_dir = os.path.join(*parent_dir.rsplit(sep))
            if parent_dir_bak.startswith(sep):
                parent_dir = os.path.join(sep, parent_dir)
                parent_dir = os.path.abspath(parent_dir)
            f = os.path.join(parent_dir, f)
        else:
            f = os.path.join('./', f)
            f = os.path.abspath(f)

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


def load_data(data_filename, **kwargs):
    # read functions arguments
    data_type = kwargs.get('data_type', 'cnv')
    logger.info('data type to process : '+data_type)

    # load data
    logger.info('load raw data: '+data_filename)
    data = pd.read_csv(data_filename, delimiter='\t', index_col=0)
    data = data.T  # samples x genes
    logger.info('data size: '+str(data.shape))
    my_bool = not(data.columns.value_counts() == 1).all()
    if my_bool:
        logger.error('gene duplicate NAMES exist!')
        return data

    return data