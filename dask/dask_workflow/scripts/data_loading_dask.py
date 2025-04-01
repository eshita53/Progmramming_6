import dask.dataframe as dd


def retrieve_data(lung_file_name,  gene_file_name):
    
    lung3_df = dd.read_csv(lung_file_name, blocksize=None) 
    gene_df = dd.read_csv(gene_file_name, sep= '\t', comment= "!", blocksize=None).compute()
    gene_df.set_index('ID_REF', inplace=True)
    gene_df = gene_df.T
    gene_df.reset_index(drop=True, inplace=True)
    combined_df = lung3_df.merge(gene_df, left_index=True, right_index=True)
    return combined_df


