
import pandas as pd

def retrieve_data(lung_file_name,  gene_file_name):
    lung3_df = pd.read_csv(lung_file_name) 
    gene_df = pd.read_csv(gene_file_name, sep= '\t', comment= "!")
    gene_df.set_index('ID_REF', inplace=True)
    gene_df = gene_df.T
    gene_df.reset_index(drop=True, inplace=True)
    combined_df = lung3_df.merge(gene_df, left_index=True, right_index=True)
    return combined_df
