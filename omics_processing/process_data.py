import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os, sys
from sklearn.model_selection import train_test_split

from utils.plotting_utils import plot_dist_and_heatmap
from utils.load_functions import load_gene_positions, load_clinical, order_patients

def reverse_preprocessing(x, mu, std, data_type='cnv'):
    print("reverse preprocessing of data to visualize them...")
    y = x.copy()
    print(" -reverse stand")
    y = (y * std) + mu
    if 'rna' in data_type:
        print(" -reverse arcsinh with sinh")
        y = pd.DataFrame(np.sinh(y.values), index = x.index, columns = x.columns)


    return y

# (train_size=0.5, toPlot = False, toSave = False, splitDir = 'split/', saveimg = False, imgpath = './')
def split_and_process_cnv(train_size=0.5, toPlot = False, toSave = False, toPrint= False,
                          rawData = 'DATA/TCGA_fromMatteo/all_processed/gistic-cn-processed.tsv',
                          outdir = 'DATA/TCGA_fromMatteo/all_processed/formatted/',
                          outname = 'cnv',
                          myseed = 0):

    # define dirs
    print('load raw data: '+rawData)
    if toSave:
        print('output will be saved in: '+outdir)
    genes_positions_table = load_gene_positions(datadir = outdir)
    imgpath = outdir+'img/'
    if toSave:
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
            print(' >created '+imgpath)

    # load CNVs from GISTIC2
    data = pd.read_csv(rawData, delimiter='\t', index_col=0)
    data = data.T
    print(' -data size: '+str(data.shape))
    my_bool = not(data.columns.value_counts() == 1).all()
    if my_bool:
        print(' -gene duplicates exist!')
        return data, None

    # check if we have more than one samples per patient
    patients = [( sample, '-'.join(sample.rsplit('-')[:3]) ) for sample in data.index.values]
    pdf = pd.DataFrame(patients)

    # extract only patient info from sample barcode
    patients = ['-'.join(sample.rsplit('-')[:3]) for sample in data.index.values]

    # replace the index
    data.insert(0,'patients',patients)
    data.set_index(['patients'], drop=True, inplace=True)
    print(' -replaced index')

    # order with grade group USING CLINICAL INFO
    clinical_table = load_clinical(data, datadir = outdir)
    mycol= 'grade_group'
    target = clinical_table[[mycol]].copy()
    data, target = order_patients(data, target)
    print('-Sorted patients by '+mycol)

    # drop rows that are ALL NaN
    print(' -data size (after sorting patients) '+str(data.shape))
    #     print('empty patients exist: '+str(data.isnull().all(1).any()))
    #     count = len(data.loc[data.isnull().all(1),:].index)
    #     print('empty patients to remove: '+str(count))
    #     data.drop(data.loc[data.isnull().all(1),:].index, axis=0, inplace=True)
    #     print('removed successfully: '+str(not data.isnull().all(1).any()))
    #     print('new CNV size '+str(data.shape))
    #     print('any NaN values in the table: '+str(data.isnull().any().any()))

    # sort the table columns with the genes position
    fname = outdir+'gaf.json'
    with open(fname, 'r') as fp:
        gene_dict = json.load(fp)
    print(' -loaded gaf file: '+fname)
    data = pd.DataFrame(data, columns=sorted(gene_dict, key=gene_dict.get))
    print('-Sorted genes by gaf')

    # drop columns that are ALL NaN
    print(' -data size (after sorting genes) '+str(data.shape))
    print(' -empty genes exist: '+str(data.isnull().all(0).any()))
    count = len(data.loc[:,data.isnull().all(0)].columns)
    print(' --empty genes to remove: '+str(count))
    data.drop(data.loc[:,data.isnull().all(0)].columns, axis=1, inplace=True)
    print(' --removed successfully: '+str(not data.isnull().all(0).any()))
    print(' -correct data size '+str(data.shape))
    print(' -any NaN values in the table: '+str(data.isnull().any().any()))


    if train_size != 0:
        if train_size < 1:
            splitname = '__split'+str(int(train_size*100))+'perc_s'+str(myseed)
        else:
            splitname = '__split'+str(train_size)+'_s'+str(myseed)
        outname = outname+splitname

        # split the data into training and testing sets
        print('-Split the data')
        X_sa, X_sb, y_sa, y_sb = train_test_split(data, target,
                                                            train_size=train_size,
                                                            random_state=myseed,
                                                            stratify= target)
        print(' -- split a data size '+str(X_sa.shape))
        print(' -- split b data size '+str(X_sb.shape))


        print('-Sort patients in split sets')
        X_sa, y_sa = order_patients(X_sa, y_sa)
        X_sb, y_sb = order_patients(X_sb, y_sb)


        if toPlot:
            plot_dist_and_heatmap(X_sa,genes_positions_table, title='cnv split a ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa1')
            plot_dist_and_heatmap(X_sb,genes_positions_table, title='cnv split b ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb1')

        # standardize data
        print('-Standardizing data in split sets')
        X_sa = (X_sa - X_sa.mean()) / X_sa.std()
        X_sb = (X_sb - X_sb.mean()) / X_sb.std()

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(X_sa,genes_positions_table, title='STAND: cnv split a ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa2')
            plot_dist_and_heatmap(X_sb,genes_positions_table, title='STAND: cnv split b ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb2')

        if toSave:
            print('-Saving split data...')
            # saving data
            outname1 = outname+'_sa'
            fname = outdir+outname1
            X_sa.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

            outname2 = outname+'_sb'
            fname = outdir+outname2
            X_sb.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return (outname1,outname2), data, target, X_sa, X_sb, y_sa, y_sb

    else:
        ## PLOT
        if toPlot:
            plot_dist_and_heatmap(data, genes_positions_table, title='cnv ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw1')

        # standardize data
        print('-Standardizing data')
        data = (data - data.mean()) / data.std()

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(data,genes_positions_table, title='STAND: cnv ', isMut=False, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw2')

        if toSave:
            print('-Saving data...')
            # saving data
            fname = outdir+outname
            data.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return outname, data, target

# def split_and_process_rna(train_size=0.5, onlyTumor = True, toPlot = False, toSave = False, splitDir = 'split/', saveimg = False, imgpath = './'):
def split_and_process_rna(train_size=0.5, onlyTumor = True, toPlot = False, toSave = False, toPrint= False,
                          rawData = 'DATA/TCGA_fromMatteo/all_processed/rnaseq-processed.tsv',
                          outdir = 'DATA/TCGA_fromMatteo/all_processed/formatted/',
                          outname = 'rna',
                          myseed = 0):
    # define dirs
    print('load raw data: '+rawData)
    if toSave:
      print('output will be saved in: '+outdir)
    genes_positions_table = load_gene_positions(datadir = outdir)
    imgpath = outdir+'img/'
    if toSave:
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
            print(' >created '+imgpath)

    # load data
    data = pd.read_csv(rawData, delimiter='\t', index_col=0)
    data = data.T
    print(' -data size: '+str(data.shape))
    my_bool = not(data.columns.value_counts() == 1).all()
    if my_bool:
        print(' -gene duplicates exist!')
        return data, None

    # check if we have more than one samples per patient
    barcode_all = [( sample, '-'.join(sample.rsplit('-')[:3]), '-'.join(sample.rsplit('-')[3:]) ) for sample in data.index.values]
    pdf = pd.DataFrame(barcode_all, columns=['barcode', 'patient','sample'])
    print(' -number of all duplicated patients '+str(pdf[pdf.patient.duplicated(keep=False)].shape[0]))

    # extract patient and sample info from barcode
    patients = [ '-'.join(sample.rsplit('-')[:3]) for sample in data.index.values]
    samples  = [ '-'.join(sample.rsplit('-')[3:]) for sample in data.index.values]
    data.insert(0,'patients',patients)
    data.insert(1,'samples',samples)

    # check what sample types exist
    print(' -sample types: '+str(data['samples'].unique()))

    if onlyTumor:
        # keep only tumor samples
        sample_code ='01'
        sample_name = 'tumor'
    else:
        # keep only normal samples (solid normal)
        sample_code ='11'
        sample_name = 'normal'
    outname = outname+'_'+sample_name

    data = data[data['samples'] == sample_code].copy()
    print(' -'+sample_name+' samples size: '+str(data.shape))

    # replace the index and order with grade group
    data.drop(['samples'], axis=1, inplace=True)
    data.set_index(['patients'], drop=True, inplace=True)
    print(' -replaced index')

    # order with grade group USING CLINICAL INFO
    fname = outdir+'clinical.txt'
    clinical_table = pd.read_csv(fname, sep='\t', index_col=0, header=0)
    print(' -loaded clinical info from: '+fname)

    mycol= 'grade_group'
    target = clinical_table[['bcr_patient_barcode', mycol]].copy()
    target.set_index(['bcr_patient_barcode'], drop=True, inplace=True, )
    del target.index.name

    data, target = order_patients(data, target)
    print('-Sorted patients by '+mycol)

    # drop rows that are ALL NaN
    print(' -data size (after sorting patients) '+str(data.shape))
    #     print('empty patients exist: '+str(data.isnull().all(1).any()))
    #     count = len(data.loc[data.isnull().all(1),:].index)
    #     print('empty patients to remove: '+str(count))
    #     data.drop(data.loc[data.isnull().all(1),:].index, axis=0, inplace=True)
    #     print('removed successfully: '+str(not data.isnull().all(1).any()))
    #     print('new CNV size '+str(data.shape))
    #     print('any NaN values in the table: '+str(data.isnull().any().any()))

    # sort the table columns with the genes position
    fname = outdir+'gaf.json'
    with open(fname, 'r') as fp:
        gene_dict = json.load(fp)
    print(' -loaded gaf file: '+fname)
    data = pd.DataFrame(data, columns=sorted(gene_dict, key=gene_dict.get))
    print('-Sorted genes by gaf')

    # drop columns that are ALL NaN
    print(' -data size (after sorting genes) '+str(data.shape))
    print(' -empty genes exist: '+str(data.isnull().all(0).any()))
    count = len(data.loc[:,data.isnull().all(0)].columns)
    print(' --empty genes to remove: '+str(count))
    data.drop(data.loc[:,data.isnull().all(0)].columns, axis=1, inplace=True)
    print(' --removed successfully: '+str(not data.isnull().all(0).any()))
    print(' -correct data size '+str(data.shape))
    print(' -any NaN values in the table: '+str(data.isnull().any().any()))



    if train_size != 0:
        if train_size < 1:
            splitname = '__split'+str(int(train_size*100))+'perc_s'+str(myseed)
        else:
            splitname = '__split'+str(train_size)+'_s'+str(myseed)
        outname = outname+splitname


        # split the data into training and testing sets
        print('-Split the data')
        X_sa, X_sb, y_sa, y_sb = train_test_split(data, target,
                                                            train_size=train_size,
                                                            random_state=myseed,
                                                            stratify= target)
        print(' -- split a data size '+str(X_sa.shape))
        print(' -- split b data size '+str(X_sb.shape))


        print('-Sort patients in split sets')
        X_sa, y_sa = order_patients(X_sa, y_sa)
        X_sb, y_sb = order_patients(X_sb, y_sb)


        if toPlot:
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='rna split a ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa1')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='rna split b ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb1')

        # arcsinh data
        print('-Transforming data with arcsinh')
        X_sa.values[:] = np.arcsinh(X_sa.values)
        X_sb.values[:] = np.arcsinh(X_sb.values)

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='ARCSINH: rna split a ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa2')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='ARCSINH: rna split b ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb2')

        # standardize data
        print('-Standardizing data in split sets')
        X_sa = (X_sa - X_sa.mean()) / X_sa.std()
        X_sb = (X_sb - X_sb.mean()) / X_sb.std()
        X_sa.fillna(0, inplace=True)
        X_sb.fillna(0, inplace=True)

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='STAND: rna split a ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa3')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='STAND: rna split b ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb3')

        if toSave:
            print('-Saving split data...')
            # saving data
            outname1 = outname+'_sa'
            fname = outdir+outname1
            X_sa.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

            outname2 = outname+'_sb'
            fname = outdir+outname2
            X_sb.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return (outname1,outname2), data, target, X_sa, X_sb, y_sa, y_sb

    else:
        ## PLOT
        if toPlot:
            plot_dist_and_heatmap(data, genes_positions_table, title='rna ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw1')

        # arcsinh data
        print('Transforming data with arcsinh')
        data.values[:] = np.arcsinh(data.values)

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(data, genes_positions_table, title='ARCSINH: rna ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw2')

        # standardize data
        print('-Standardizing data')
        data = (data - data.mean()) / data.std()
        data.fillna(0, inplace=True)

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(data, genes_positions_table, title='STAND: rna ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw3')

        if toSave:
            print('-Saving data...')
            # saving data
            fname = outdir+outname
            data.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return outname, data, target



# def split_and_process_meth(train_size=0.5, onlyTumor = True, toPlot = False, toSave = False, splitDir = 'split/', saveimg = False, imgpath = './'):
def split_and_process_meth(train_size=0.5, onlyTumor = True, toPlot = False, toSave = False, toPrint= False,
                          rawData = 'DATA/TCGA_fromMatteo/all_processed/methylation-processed.tsv',
                          outdir = 'DATA/TCGA_fromMatteo/all_processed/formatted/',
                          outname = 'meth',
                          myseed = 0):
    # define dirs
    print('load raw data: '+rawData)
    if toSave:
      print('output will be saved in: '+outdir)
    genes_positions_table = load_gene_positions(datadir = outdir)
    imgpath = outdir+'img/'
    if toSave:
      if not os.path.exists(imgpath):
          os.makedirs(imgpath)
          print(' >created '+imgpath)

    # load data
    data = pd.read_csv(rawData, delimiter='\t', index_col=0)
    data = data.T
    print(' -data size: '+str(data.shape))
    my_bool = not(data.columns.value_counts() == 1).all()
    if my_bool:
        print(' -gene duplicates exist!')
        return data, None

    # check if we have more than one samples per patient
    barcode_all = [( sample, '-'.join(sample.rsplit('-')[:3]), '-'.join(sample.rsplit('-')[3:]) ) for sample in data.index.values]
    pdf = pd.DataFrame(barcode_all, columns=['barcode', 'patient','sample'])
    print(' -number of all duplicated patients '+str(pdf[pdf.patient.duplicated(keep=False)].shape[0]))

    # extract patient and sample info from barcode
    patients = [ '-'.join(sample.rsplit('-')[:3]) for sample in data.index.values]
    samples  = [ '-'.join(sample.rsplit('-')[3:]) for sample in data.index.values]
    data.insert(0,'patients',patients)
    data.insert(1,'samples',samples)

    # check what sample types exist
    print(' -sample types: '+str(data['samples'].unique()))

    if onlyTumor:
        # keep only tumor samples
        sample_code ='01'
        sample_name = 'tumor'
    else:
        # keep only normal samples (solid normal)
        sample_code ='11'
        sample_name = 'normal'
    outname = outname+'_'+sample_name

    data = data[data['samples'] == sample_code].copy()
    print(' -'+sample_name+' samples size: '+str(data.shape))

    # replace the index and order with grade group
    data.drop(['samples'], axis=1, inplace=True)
    data.set_index(['patients'], drop=True, inplace=True)
    print(' -replaced index')

    # order with grade group USING CLINICAL INFO
    clinical_table = load_clinical(data, datadir = outdir)
    mycol= 'grade_group'
    target = clinical_table[[mycol]].copy()
    data, target = order_patients(data, target)
    print('-Sorted patients by '+mycol)

    # drop rows that are ALL NaN
    print(' -data size (after sorting patients) '+str(data.shape))
    #     print('empty patients exist: '+str(data.isnull().all(1).any()))
    #     count = len(data.loc[data.isnull().all(1),:].index)
    #     print('empty patients to remove: '+str(count))
    #     data.drop(data.loc[data.isnull().all(1),:].index, axis=0, inplace=True))
    #     print('removed successfully: '+str(not data.isnull().all(1).any()))
    #     print('new CNV size '+str(data.shape))
    #     print('any NaN values in the table: '+str(data.isnull().any().any()))


    nan_patients = data.isnull().sum(axis=1)
    nan_genes = data.isnull().sum(axis=0)

    print(' --all NaN values: '+str(data.isnull().sum().sum())+' (perc = '+str(100*round(data.isnull().sum().sum()/float(data.size),6))+'%)')
    print(' --patients with NaN values: '+str(nan_patients[nan_patients>0].shape[0])+' (perc = '+str(100*round(nan_patients[nan_patients>0].shape[0]/float(nan_patients.shape[0]),4))+'%)')
    print(' --patients with more than one NaN values: '+str(nan_patients[nan_patients>1].shape[0])+' (perc = '+str(100*round(nan_patients[nan_patients>1].shape[0]/float(nan_patients.shape[0]),4))+'%)')
    print(' --genes with NaN values: '+str(nan_genes[nan_genes>0].shape[0])+' (perc = '+str(100*round(nan_genes[nan_genes>0].shape[0]/float(nan_genes.shape[0]),4))+'%)')
    print(' --genes with more than one NaN values: '+str(nan_genes[nan_genes>1].shape[0])+' (perc = '+str(100*round(nan_genes[nan_genes>1].shape[0]/float(nan_genes.shape[0]),4))+'%)')

    data.fillna(data.mean(), inplace=True)
    print('-Set NaN to column/gene mean.')


    # sort the table columns with the genes position
    fname = outdir+'gaf.json'
    with open(fname, 'r') as fp:
        gene_dict = json.load(fp)
    print(' -loaded gaf file: '+fname)
    data = pd.DataFrame(data, columns=sorted(gene_dict, key=gene_dict.get))
    print('-Sorted genes by gaf')

    # drop columns that are ALL NaN
    print(' -data size (after sorting genes) '+str(data.shape))
    print(' -empty genes exist: '+str(data.isnull().all(0).any()))
    count = len(data.loc[:,data.isnull().all(0)].columns)
    print(' --empty genes to remove: '+str(count))
    data.drop(data.loc[:,data.isnull().all(0)].columns, axis=1, inplace=True)
    print(' --removed successfully: '+str(not data.isnull().all(0).any()))
    print(' -correct data size '+str(data.shape))
    print(' -any NaN values in the table: '+str(data.isnull().any().any()))



    if train_size != 0:
        if train_size < 1:
            splitname = '__split'+str(int(train_size*100))+'perc_s'+str(myseed)
        else:
            splitname = '__split'+str(train_size)+'_s'+str(myseed)
        outname = outname+splitname


        # split the data into training and testing sets
        print('-Split the data')
        X_sa, X_sb, y_sa, y_sb = train_test_split(data, target,
                                                            train_size=train_size,
                                                            random_state=myseed,
                                                            stratify= target)
        print(' -- split a data size '+str(X_sa.shape))
        print(' -- split b data size '+str(X_sb.shape))


        print('-Sort patients in split sets')
        X_sa, y_sa = order_patients(X_sa, y_sa)
        X_sb, y_sb = order_patients(X_sb, y_sb)


        if toPlot:
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='meth split a ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa1')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='meth split b ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb1')

        # standardize data
        print('-Standardizing data in split sets')
        X_sa = (X_sa - X_sa.mean()) / X_sa.std()
        X_sb = (X_sb - X_sb.mean()) / X_sb.std()

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='STAND: meth split a ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa2')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='STAND: meth split b ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb2')

        if toSave:
            print('-Saving split data...')
            # saving data
            outname1 = outname+'_sa'
            fname = outdir+outname1
            X_sa.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

            outname2 = outname+'_sb'
            fname = outdir+outname2
            X_sb.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return (outname1,outname2), data, target, X_sa, X_sb, y_sa, y_sb

    else:
        ## PLOT
        if toPlot:
            plot_dist_and_heatmap(data, genes_positions_table, title='meth ', isMut=False, cmap = 'pos', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw1')

        # standardize data
        print('-Standardizing data')
        data = (data - data.mean()) / data.std()

        if toPlot:
            # show data values distribution and heatmaps
            plot_dist_and_heatmap(data, genes_positions_table, title='STAND: meth ', isMut=False, cmap = 'div', saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw2')

        if toSave:
            print('-Saving data...')
            # saving data
            fname = outdir+outname
            data.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return outname, data, target


# def split_and_process_mut(train_size=0.5, toPlot = False, toSave = False, splitDir = 'split/', saveimg = False, imgpath = './'):
def split_and_process_mut(train_size=0.5, toPlot = False, toSave = False, toPrint= False,
                          rawData = 'DATA/TCGA_fromMatteo/all_processed/maf-processed.tsv',
                          outdir = 'DATA/TCGA_fromMatteo/all_processed/formatted/',
                          outname = 'mut',
                          myseed = 0):
    # define dirs
    print('load raw data: '+rawData)
    if toSave:
      print('output will be saved in: '+outdir)
    genes_positions_table = load_gene_positions(datadir = outdir)
    imgpath = outdir+'img/'
    if toSave:
      if not os.path.exists(imgpath):
          os.makedirs(imgpath)
          print(' >created '+imgpath)

    # load data
    data = pd.read_csv(rawData, delimiter='\t', index_col=0)
    data = data.T
    print(' -data size: '+str(data.shape))
    my_bool = not(data.columns.value_counts() == 1).all()
    if my_bool:
        print(' -gene duplicates exist!')
        return data, None

    # check if we have more than one samples per patient
    barcode_all = [( sample, '-'.join(sample.rsplit('-')[:3]), '-'.join(sample.rsplit('-')[3:]) ) for sample in data.index.values]
    pdf = pd.DataFrame(barcode_all, columns=['barcode', 'patient','sample'])
    print(' -number of all duplicated patients '+str(pdf[pdf.patient.duplicated(keep=False)].shape[0]))

    # extract patient and sample info from barcode
    patients = [ '-'.join(sample.rsplit('-')[:3]) for sample in data.index.values]
    samples  = [ '-'.join(sample.rsplit('-')[3:]) for sample in data.index.values]
    data.insert(0,'patients',patients)
    data.insert(1,'samples',samples)

    # check what sample types exist
    print(' -sample types: '+str(data['samples'].unique()))


    # replace the index and order with grade group
    data.drop(['samples'], axis=1, inplace=True)
    data.set_index(['patients'], drop=True, inplace=True)
    print(' -replaced index')

    # order with grade group USING CLINICAL INFO
    fname = outdir+'clinical.txt'
    clinical_table = pd.read_csv(fname, sep='\t', index_col=0, header=0)
    print(' -loaded clinical info from: '+fname)

    mycol= 'grade_group'
    target = clinical_table[['bcr_patient_barcode', mycol]].copy()
    target.set_index(['bcr_patient_barcode'], drop=True, inplace=True, )
    del target.index.name

    data, target = order_patients(data, target)
    print('-Sorted patients by '+mycol)

    # drop rows that are ALL NaN
    print(' -data size (after sorting patients) '+str(data.shape))
    #     print('empty patients exist: '+str(data.isnull().all(1).any()))
    #     count = len(data.loc[data.isnull().all(1),:].index)
    #     print('empty patients to remove: '+str(count))
    #     data.drop(data.loc[data.isnull().all(1),:].index, axis=0, inplace=True)
    #     print('removed successfully: '+str(not data.isnull().all(1).any()))
    #     print('new CNV size '+str(data.shape))
    #     print('any NaN values in the table: '+str(data.isnull().any().any()))

    # sort the table columns with the genes position
    fname = outdir+'gaf.json'
    with open(fname, 'r') as fp:
        gene_dict = json.load(fp)
    print(' -loaded gaf file: '+fname)
    data = pd.DataFrame(data, columns=sorted(gene_dict, key=gene_dict.get))
    print('-Sorted genes by gaf')

    # drop columns that are ALL NaN
    print(' -data size (after sorting genes) '+str(data.shape))
    print(' -empty genes exist: '+str(data.isnull().all(0).any()))
    count = len(data.loc[:,data.isnull().all(0)].columns)
    print(' --empty genes to remove: '+str(count))
    data.drop(data.loc[:,data.isnull().all(0)].columns, axis=1, inplace=True)
    print(' --removed successfully: '+str(not data.isnull().all(0).any()))
    print(' -correct data size '+str(data.shape))
    print(' -any NaN values in the table: '+str(data.isnull().any().any()))

    if train_size != 0:
        # remove hypermutated samples
        tmp = data.shape[0]
        data = data[data.sum(axis =1) < data.sum(axis=1).mean()+data.sum(axis=1).std()]
        print('-Remove '+str(tmp - data.shape[0])+' hypermutated samples')
        data, target = order_patients(data, target)
        print('-Sorted patients by '+mycol)

        if train_size < 1:
            splitname = '__split'+str(int(train_size*100))+'perc_s'+str(myseed)
        else:
            splitname = '__split'+str(train_size)+'_s'+str(myseed)
        outname = outname+splitname


        # split the data into training and testing sets
        print('-Split the data')
        X_sa, X_sb, y_sa, y_sb = train_test_split(data, target,
                                                            train_size=train_size,
                                                            random_state=myseed,
                                                            stratify= target)
        print(' -- split a data size '+str(X_sa.shape))
        print(' -- split b data size '+str(X_sb.shape))


        print('-Sort patients in split sets')
        X_sa, y_sa = order_patients(X_sa, y_sa)
        X_sb, y_sb = order_patients(X_sb, y_sb)

        if toPlot:
            plot_dist_and_heatmap(X_sa, genes_positions_table, title='mut split a ', isMut=True, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sa1')
            plot_dist_and_heatmap(X_sb, genes_positions_table, title='mut split b ', isMut=True, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_sb1')

        if toSave:
            print('-Saving split data...')
            # saving data
            outname1 = outname+'_sa'
            fname = outdir+outname1
            X_sa.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

            outname2 = outname+'_sb'
            fname = outdir+outname2
            X_sb.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return (outname1,outname2), data, target, X_sa, X_sb, y_sa, y_sb

    else:
        ## PLOT
        if toPlot:
            plot_dist_and_heatmap(data, genes_positions_table, title='mut ', isMut=True, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw1')

        if toSave:
            print('-Saving data (with hypermutated samples)...')
            # saving data
            fname = outdir+outname+'_wHyperMut.txt'
            data.to_csv(fname, sep='\t')
            print(' -saved: '+fname)

        # remove hypermutated samples
        tmp = data.shape[0]
        data = data[data.sum(axis =1) < data.sum(axis=1).mean()+data.sum(axis=1).std()]
        print('-Remove '+str(tmp - data.shape[0])+' hypermutated samples')
        data, target = order_patients(data, target)
        print('-Sorted patients by '+mycol)

        if toPlot:
            # show mut values distribution and heatmaps
            plot_dist_and_heatmap(data, genes_positions_table, title='mut ', isMut=True, saveimg = toSave, showimg = toPrint, imgpath = imgpath+outname+'_raw1new')

        if toSave:
            print('-Saving data...')
            # saving data
            fname = outdir+outname
            data.to_csv(fname+'.txt', sep='\t')
            print(' -saved: '+fname)

        return outname, data, target
