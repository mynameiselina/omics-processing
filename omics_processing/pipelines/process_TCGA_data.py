"""
process the CNV data from TCGA PRAD
load_data, clean_samples, split_data,
clean_genes, transform_data, sort_data

clean_samples:
check if we have more than one samples per patient
extract patient and sample info from barcode
check what sample types exist
replace the index

split_data:
split the data into training and testing sets
(train_size should be smaller than 1.0 or be an integer)

clean_genes:
drop columns that are ALL NaN
sanity check (TODO: set imputation in case it fails)
drop columns that contain all zeroes

transform_data:
arcsinh transformation [optional]
standardize transformation

sort_data:
sort the table columns with the genes chr position
sort samples by 'grade_group'
"""

# custom imports
from omics_processing.io import (
    load_gene_order_dict, load_clinical,
    join_path, fpath_absolute_relative
)
from omics_processing.process_data import (
    load_data, split_data, clean_samples, clean_genes,
    transform_data, sort_data, save_output
)
from omics_processing.remove_duplicates import remove_andSave_duplicates

# basic imports
import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)

script_path = os.path.dirname(__file__)


def run_pipeline(
    # load_data
    filepath="TCGA_PRAD/input/gistic-cn-processed.tsv",
    clinical_fpath="TCGA_PRAD/processed/clinical.txt",
    gaf_fpath="TCGA_PRAD/processed/gaf.json",
    output_directory="TCGA_PRAD/processed",
    data_type='cnv',
    # clean_samples
    sample_type=None,
    # split_data
    split_train_size=None,
    split_random_state=None,
    stratify_patients_by=None,
    # transform_data
    to_arcsinh=False,
    to_stand=True,
    # sort_data: genes
    to_sort_columns=True,
    gene_id_col='gene',
    gene_order_col='pos_order',
    # sort_data: samples
    to_sort_rows=True,
    sort_patients_by="grade_group",
    # remove_andSave_duplicates
    to_remove_duplicate_columns=True,
    to_compute_euclidean_distances=True,
    to_save_euclidean_distances=True,
    # final output main name
    output_filename=None
):
    # set main data directory
    MainDataDir = join_path(script_path+'/../../data')

    # make fpaths valid
    # data input
    filepath = fpath_absolute_relative(
        filepath, rootDir=MainDataDir, name="data input")
    # gene order table
    gaf_fpath = fpath_absolute_relative(
        gaf_fpath, rootDir=MainDataDir, name="gene order table")
    # clinical info
    clinical_fpath = fpath_absolute_relative(
        clinical_fpath, rootDir=MainDataDir, name="clinical info")
    clinical = load_clinical(
        clinical_fpath,
        **{
            'sep': '\t',
            'header': 0,
            'index_col': 0
        })
    # output dir
    output_directory = os.path.join(MainDataDir, join_path(output_directory))

    if output_filename is None:
        output_filename = ''

    # load_data, clean_samples, split_data,
    # clean_genes, transform_data, sort_data
    data = load_data(filepath, data_type=data_type)
    output_filename = output_filename+data_type
    logger.debug("finished loading data")

    # clean genes
    data = clean_genes(data)
    logger.debug("finished cleaning genes in all samples")

    if to_sort_columns:
        genes_positions_table = pd.read_csv(
            gaf_fpath, sep='\t', header=0, index_col=0)
        gene_order = genes_positions_table.set_index(
            gene_id_col).loc[:, gene_order_col]
        logger.debug(
            "finished loading genes order")
        data = sort_data(
            data, to_sort_columns=True,
            to_sort_rows=False,
            gene_order=gene_order)
        logger.debug("finished sorting genes in all samples")

    data = clean_samples(data, sample_type=sample_type)
    logger.debug("finished cleaning samples")
    # in case multiple sample types exist,
    # the 'sample_type' will change inside clean_data()
    if sample_type is not None:
        output_filename = output_filename+'_'+sample_type

    if split_train_size is not None:
        stratify_patients_by = stratify_patients_by.rsplit(',')
        stratify_by = \
            clinical.loc[data.index, stratify_patients_by]
        data_list = split_data(
            data, stratify_by=stratify_by,
            split_train_size=split_train_size,
            split_random_state=split_random_state)
        logger.debug("finished splitting data sample sets")
        if split_train_size < 1:
            splitname = '_split_perc' + \
                        str(int(split_train_size*100)) + \
                        '_seed'+str(split_random_state)
        else:
            splitname = '_split_size' + \
                        str(int(split_train_size)) + \
                        '_seed'+str(split_random_state)
        output_filename = output_filename+'_'+splitname
        fname_list = [output_filename+'_part1', output_filename+'_part2']
    else:
        data_list = [data]
        fname_list = [output_filename]

    _counter = 0
    for _data, _fname in zip(data_list, fname_list):

        _data = clean_genes(_data)
        logger.debug("finished cleaning genes in set "+str(_counter))

        _data, transformation_settings = transform_data(
            _data, to_arcsinh=to_arcsinh, to_stand=to_stand)
        logger.debug("finished transforming data in set "+str(_counter))

        if to_sort_rows:
            sort_patients_by = sort_patients_by.rsplit(',')
            logger.debug("sorting samples in set "+str(_counter))
            _data = sort_data(
                _data, to_sort_columns=False,
                to_sort_rows=True,
                sort_patients_by=sort_patients_by,
                clinical=clinical)
            logger.debug("finished sorting samples in set "+str(_counter))

        output_filename = _fname+'_processed'
        save_output(
            _data, transformation_settings=transformation_settings,
            output_directory=output_directory, output_filename=output_filename)
        logger.debug("finished saving data output in set "+str(_counter))

        if to_remove_duplicate_columns:
            load_from_file = False
            to_save_output = True
            newdata, _, _, _ = remove_andSave_duplicates(
                _data, load_from_file=load_from_file,
                to_compute_euclidean_distances=to_compute_euclidean_distances,
                to_save_euclidean_distances=to_save_euclidean_distances,
                to_save_output=to_save_output,
                output_filename=output_filename,
                output_directory=output_directory)
            logger.debug(
                "finished removing duplicate gene profiles in set " +
                str(_counter))

            # sanity check (TODO: set imputation in case it fails)
            nan_exist = newdata.isnull().any().any()
            if nan_exist:
                logger.error(
                    'NaN values exist in the newdata!\nNo imputation is set!')
                raise

        _counter += 1
