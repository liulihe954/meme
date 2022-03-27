import argparse, os, sys
import numpy as np
# You can choose to write classes in other python files
# and import them here.
from scipy import stats
import pandas as pd

###
def to_bin_raw_data(filepath, bin_n, bin_type):
    ''' Function to bin the raw data given number of bins and type of bins
    Args:
        bin_n: the number of bins to use
        bin_type: the type of bins, either "size" or "density"
    '''
    # read in raw data and get dimension
    with open(filepath, 'r') as f:
        raw_data = pd.read_table(f).drop("Time",axis=1).values
        n_time, n_gene = raw_data.shape[0], raw_data.shape[1]
        data_after_bin = np.zeros((n_time, n_gene)) # container for binned data
    # 
    if bin_type.upper() == "UNIFORM":
        for i in range(n_gene):
            gene_min_expr_temp = min(raw_data[:,i])
            gene_max_expr_temp = max(raw_data[:,i])
            bin_width = (gene_max_expr_temp - gene_min_expr_temp)/bin_n
            bin_arr = []
            for j in range(1, bin_n):
                bin_arr.append(j * bin_width + gene_min_expr_temp)
            data_after_bin[:,i] = np.digitize(raw_data[:,i], bin_arr, right = True)
    elif bin_type.upper() == "DENSITY":
        for i in range(n_gene):
            bin_perct_unit = 100/bin_n
            bin_arr = []
            for j in range(1, bin_n):
                bin_arr.append(np.percentile(raw_data[:,i], j * bin_perct_unit))
            data_after_bin[:,i] = np.digitize(raw_data[:,i], bin_arr, right = True)
    else:
        raise Exception("Invalid bin_str, should be either 'size' or 'density'")
        
    return (data_after_bin.astype(int), bin_n, n_gene, n_time)

###
def processNoutput(bin_data):
    ''' Function to process and sort the output data
    Args:
        bin_data: the pre-processed data (reutn from the function to_bin_raw_data)
    '''
    # parsing input 
    data_bin = bin_data[0]
    bin_n = bin_data[1]
    n_genes = bin_data[2]
    n_time = bin_data[3]
    
    # set container
    mi_output = {}
    mi_temp = np.zeros((n_genes, n_genes))
    binXgene_count = np.zeros((bin_n, n_genes))
    
    # count: bin by gene
    for i in range(n_time):
        for j in range(n_genes):
            binXgene_count[data_bin[i,j],j] += 1
    binXgene_count += 0.1 
    
    # pair wise calcuation of mi
    for i in range(n_genes):
        for j in range(1+i, n_genes):
            #
            pairwise_count = np.zeros((bin_n, bin_n))
            for k in range(n_time):
                pairwise_count[data_bin[k,i], data_bin[k,j]] += 1
            pairwise_count += 0.1
            #
            prob_ij = pairwise_count/np.sum(pairwise_count)
            prob_i = binXgene_count[:,i]/np.sum(binXgene_count[:,i])
            prob_j = binXgene_count[:,j]/np.sum(binXgene_count[:,j])
            
            for m in range(bin_n):
                for n in range(bin_n):
                    mi_temp[i,j] += prob_ij[m,n] * np.log2(prob_ij[m,n]/(prob_i[m] * prob_j[n]))
            mi_output[(i+1,j+1)] = np.round(mi_temp[i,j], decimals=3)
    
    # sort and tie breaking
    return sorted(sorted(mi_output.items(),
                         key=lambda x: (x[0][0], x[0][1])),
                  key=lambda x: x[1],
                  reverse=True)

# This is the main function provided for you.
# Define additional functions to implement mutual information
def main(args):
    # Parse input arguments
    # It is generally good practice to validate the input arguments, e.g.,
    # verify that the input and output filenames are provided
    data_file_path = args.dataset
    bin_num = args.bin_num
    bin_str = args.bin_str
    output_file_path = args.out

    # Where you run your code.
    
    # process data
    bin_data = to_bin_raw_data(data_file_path, bin_num, bin_str)
    output_data = processNoutput(bin_data)
    
    # output file
    with open(output_file_path,'w') as fout:
        for b,v in output_data:
            print("({},{})\t{:.3f}".format(b[0]+1,b[1]+1,v), file=fout)
    
# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
    # Note: this example shows named command line arguments.  See the argparse
    # documentation for positional arguments and other examples.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset',
                        help='input gene expression data file path',
                        type=str)
    parser.add_argument('--bin_num',
                        help='number of bins',
                        type=int,
                        default=5)
    parser.add_argument('--bin_str',
                        help='binning strategy',
                        type=str,
                        choices={'uniform', 'density'},
                        default='uniform')
    parser.add_argument('--out',
                        help='MI output file path',
                        type=str,
                        default='uniform.txt')

    args = parser.parse_args()
    # Note: this simply calls the main function above, which we could have
    # given any name
    main(args)