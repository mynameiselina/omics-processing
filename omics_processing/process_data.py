import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from .io import set_directory

script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


# order with grade group USING CLINICAL INFO
def _sort_patients(data, properties, sort_by=['grade_group']):
    # keep only patients that exist in the data table
    properties_sorted = properties.reindex(data.index)

    # sort clinical (in case it was not already sorted)
    properties_sorted.sort_values(sort_by, ascending=[0], inplace=True)

    # now order the data table rows according to clinical order
    # (and drop the ones that do not exist)
    data_sorted = data.reindex(properties_sorted.index)

    return data_sorted, properties_sorted


def clean_samples(data, **kwargs):
    logger.info('clean samples: dupl, sample info, index')
    # check if we have more than one samples per patient
    barcode_all = [(sample, '-'.join(sample.rsplit('-')[:3]),
                    '-'.join(sample.rsplit('-')[3:]))
                   for sample in data.index.values]
    barcode_parts = pd.DataFrame(barcode_all,
                                 columns=['barcode', 'patient', 'sample'])
    logger.info('number of all duplicated patients: ' +
                str(barcode_parts[
                    barcode_parts['patient'].duplicated(keep=False)
                    ].shape[0]))

    # extract patient and sample info from barcode
    patients = ['-'.join(sample.rsplit('-')[:3])
                for sample in data.index.values]
    samples = ['-'.join(sample.rsplit('-')[3:])
               for sample in data.index.values]
    data.insert(0, 'patients', patients)
    data['patients'] = data['patients'].str.lower()
    data.insert(1, 'samples', samples)

    # check what sample types exist
    uniq_sample_types = data['samples'].unique()
    logger.info('sample types: '+str(uniq_sample_types))

    # if multiple sample types exist
    if len(uniq_sample_types) > 1:
        logger.info('multiple sample types in the data')
        sample_type = kwargs.get('sample_type', 'tumor')
        if sample_type == 'tumor':
            # keep only tumor samples
            sample_code = '01'
        elif sample_type == 'normal':
            # keep only normal samples (solid normal)
            sample_code = '11'
        logger.info(sample_type+' sample type chosen')

        data = data[data['samples'] == sample_code].copy()
        logger.info(sample_type+' samples size: '+str(data.shape))

    # replace the index
    data.drop(['samples'], axis=1, inplace=True)
    data.set_index(['patients'], drop=True, inplace=True)
    logger.info('replaced index')

    return data


def clean_genes(data):
    logger.info('clean genes: remove empty and zero columns')
    # drop columns that are ALL NaN
    null_columns = data.columns.values[data.isnull().all(0)]
    logger.info('null genes to remove: '+str(null_columns.shape[0]))
    if null_columns.shape[0] > 0:
        data.drop(null_columns, axis=1, inplace=True)
        logger.info('data shape (after removing null genes): '+str(data.shape))

    # sanity check (TODO: set imputation in case it fails)
    nan_exist = data.isnull().any().any()
    if nan_exist:
        nan_sum = data.isnull().sum().sum()
        nan_perc = np.round(100*nan_sum / float(data.size), 2)
        logger.warning(
            'NaN values exist in the data: ' +
            str(nan_perc)+'% is NaN!\n' +
            'No imputation method is set, NaN will be set to zero.')
        data = data.fillna(0)

    # drop columns that contain all zeroes
    zero_columns = data.columns.values[np.where(abs(data).sum(axis=0) == 0)[0]]
    logger.info('zero genes to remove: '+str(zero_columns.shape[0]))
    if zero_columns.shape[0] > 0:
        data.drop(zero_columns, axis=1, inplace=True)
        logger.info('data shape (after removing zero genes): '+str(data.shape))

    return data


def reverse_processing(x, settings):
    # works for data that have been only standardized
    # of first arcsinh and then standardized
    settings = settings.loc[x.columns, :].copy()
    if 'stand' in settings.columns.values:
        if settings['stand'].all():
            mu = settings['mean'].copy()
            std = settings['std'].copy()
            logger.info("reverse standardization of columns")
            x = (x * std) + mu

            if 'arcsinh' in settings.columns.values:
                if settings['arcsinh'].all():
                    x = pd.DataFrame(
                        np.sinh(x.values), index=x.index, columns=x.columns)
    return x


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


def split_data(data, **kwargs):
    stratify_by = kwargs.get('stratify_by', None)
    split_train_size = kwargs.get('split_train_size', 0.5)
    split_random_state = kwargs.get('split_random_state', 0)

    # train_size should be smaller than 1.0 or be an integer
    if split_train_size >= 1:
        split_train_size = int(split_train_size)
    # split the data into training and testing sets
    logger.info('Splitting data...')
    if stratify_by is None:
        data_s1, data_s2 = \
            train_test_split(data,
                             train_size=split_train_size,
                             # From version 0.21, test_size will always
                             # complement train_size unless both are specified.
                             test_size=None,
                             random_state=split_random_state)
    else:
        data_s1, data_s2, y_s1, y_s2 = \
            train_test_split(data, stratify_by,
                             train_size=split_train_size,
                             # From version 0.21, test_size will always
                             # complement train_size unless both are specified.
                             test_size=None,
                             random_state=split_random_state,
                             stratify=stratify_by)
    logger.info('data, split 1, size: '+str(data_s1.shape))
    logger.info('data, split 2, size: '+str(data_s2.shape))

    return data_s1, data_s2


def transform_data(data, **kwargs):
    transformation_settings = \
        pd.DataFrame(index=data.columns,
                     columns=['arcsinh', 'stand', 'mean', 'std'])
    to_arcsinh = kwargs.get('to_arcsinh', False)
    transformation_settings['arcsinh'] = to_arcsinh
    to_stand = kwargs.get('to_stand', True)
    transformation_settings['stand'] = to_stand

    # arcsinh transformation
    if to_arcsinh:
        logger.info('transformation: arcsinh data...')
        data.values[:] = np.arcsinh(data.values)

    # standardize transformation
    if to_stand:
        logger.info('transformation: standardize data...')
        data_mean = data.mean()
        data_std = data.std()
        data = (data - data_mean) / data_std
        transformation_settings['mean'] = data_mean
        transformation_settings['std'] = data_std

    # sanity check (TODO: set imputation in case it fails)
    nan_exist = data.isnull().any().any()
    if nan_exist:
        logger.error('NaN values exist in the data!\nNo imputation is set!')
        raise

    return data, transformation_settings


def sort_data(data, **kwargs):
    to_sort_columns = kwargs.get('to_sort_columns', False)
    to_sort_rows = kwargs.get('to_sort_rows', False)

    # sort the table columns with the genes position
    if to_sort_columns:
        gene_order = kwargs.get('gene_order', None)
        # # extract the gene relative order
        # gene_order = genes_positions_table.set_index(
        #     gene_id_col).loc[:, 'order'].copy()
        if gene_order is None:
            logger.error('gene_order is missing: ' +
                         'genes will not be sorted')
            raise
        # keep only gene_order with data
        ids_tmp = set(
            gene_order.index.values).intersection(set(data.columns.values))
        # keep only the order of these genes
        gene_order = gene_order.loc[ids_tmp].copy()
        gene_order = gene_order.sort_values()
        # then keep only these genes from the data
        data = data.loc[:, gene_order.index].copy()
        logger.info('sorted genes by chromosome position with gene_order')
        logger.info('data shape: '+str(data.shape))

    # sort samples by 'grade_group'
    if to_sort_rows:
        sort_patients_by = kwargs.get('sort_patients_by', ['grade_group'])
        clinical = kwargs.get('clinical', None)
        if clinical is None:
            logger.error('clinical data is missing: ' +
                         'samples will not be sorted')
            raise

        data, clinical = _sort_patients(data, clinical, sort_patients_by)
        logger.info('sorted patients by: '+str(sort_patients_by))
        logger.info('data shape: '+str(data.shape))

    return data


def save_output(data, **kwargs):
    output_directory = kwargs.get('output_directory',
                                  os.path.join(script_path, "..", "data",
                                               "processed"))
    output_filename = kwargs.get('output_filename', 'data_processed.txt')
    transformation_settings = kwargs.get('transformation_settings', None)

    # define dirs
    logger.info('output files will be saved in: '+output_directory)
    _ = set_directory(output_directory)

    # saving data (csv)
    fpath = os.path.join(output_directory, output_filename)
    logger.info('save data as csv: '+fpath+'.txt')
    data.to_csv(fpath+'.txt', sep='\t')

    # saving transformation_settings (csv)
    if transformation_settings is not None:
        fpath = fpath+'__transformation_settings'
        logger.info('save transformation_settings as csv: '+fpath+'.txt')
        transformation_settings.to_csv(fpath+'.txt', sep='\t')
