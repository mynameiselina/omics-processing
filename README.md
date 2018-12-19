# Processing pipelines for -omic datasets

## Dependencies

- Python 3.6.3
- Numpy (>= 1.15)
- Pandas (>= 0.23)
- SciPy (>= 1.1.0)
- Scikit-Learn (>=0.19.2)
- Plac (>=1.0.0)
- H5py (>=2.8.0)
- Natsort (>=5.1.0)

## Installation
Install `omic-processing` after cloning:

```sh
pip3 install .
```
For developers is better to create an editable installation (symbolic links):

```sh
pip3 install -e .
```

## Example
To see how to implement the `omic-processing` main pipeline for the TCGA data please checkout out the pipelines/process_TCGA_data.py script.

## Command line usage
```sh
process_TCGA_gaf 
    -filepath /path_to_gaf/gene.genome.v4_0.processed.gaf
    -outdir /path_to_output
    [-D]
```
```sh
process_TCGA_clinical 
    -filepath /path_to_clinical/gdac.broadinstitute.org_PRAD.Merge_Clinical.Level_1.2016012800.0.0/PRAD.clin.merged.txt
    -outdir /path_to_output
    [-D]
```
```sh
process_TCGA_data -config path_to_json_config/config.json [-D]
```
The commands have been created to process the default files from TCGA and will not work on other formats.

The config file should set the parameters to process TCGA data, examples can be found in the examples/configs directory.
Optionally all these parameters could also be set from command line. If the same parameter has been set in command line and in the config file then the one from the command line will be chosen.
The -D option is to set the debug mode on and have a more verbose printout.
