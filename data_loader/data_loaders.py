import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from os.path import join, exists
from scipy import sparse
from torch.utils.data import Dataset
from base import BaseDataLoader

class ddiDataset(Dataset):
    def __init__(self, ddi_df):
        self.ddi_array = ddi_df.values
    
    def __len__(self):
        return len(self.ddi_array)
    
    def __getitem__(self, index):
        drug_1, drug_2, label = self.ddi_array[index]
        drug_1 = torch.from_numpy(np.array([drug_1])).type(torch.LongTensor)
        drug_2 = torch.from_numpy(np.array([drug_2])).type(torch.LongTensor)
        label = torch.from_numpy(np.array([label])).type(torch.FloatTensor)

        return drug_1, drug_2, label


class DDIGraphDataLoader(BaseDataLoader):
    def __init__(self, logger, data_dir, batch_size, shuffle=True, validation_split=0.1, test_split=0.2, num_workers=1):
        self.logger = logger
        self.data_dir = data_dir
        self.ddi_df, self.dpi_df, self.ppi_df = self._load_data()
        self.graph = self._build_graph()
        self.node_num = self.graph.number_of_nodes()
        self.adj_tensor = self._get_sparse_adj()
        dataset = self._create_dataset()
        super().__init__(dataset, batch_size, shuffle, validation_split, test_split, num_workers)
    
    def get_adj(self):
        return self.adj_tensor
    
    def get_node_num(self):
        return self.graph.number_of_nodes()

    def _load_data(self):
        self.logger.info('Loading data...')
        if exists(join(self.data_dir, 'processed')):
            ddi_df = pd.read_csv(join(self.data_dir, 'processed', 'ddi.csv'))
            dpi_df = pd.read_csv(join(self.data_dir, 'processed', 'dpi.csv'))
            ppi_df = pd.read_csv(join(self.data_dir, 'processed', 'ppi.csv'))
            return ddi_df, dpi_df, ppi_df
        else:
            os.makedirs(join(self.data_dir, 'processed'), exist_ok=True)
        
        self.ddi_df = pd.read_csv(join(self.data_dir, 'ddi.csv'), 
            usecols=['drug1', 'drug2', 'type'])
        self.ddi_df.columns = ['drug_1', 'drug_2', 'label']

        self.dpi_df = pd.read_csv(join(self.data_dir, 'dpi.csv'),
            usecols=['drugbank_id', 'entrez_id'])
        self.dpi_df.columns = ['drug', 'protein']
        self.tcm_target_df = pd.read_csv(join(self.data_dir, 'tcm', 'tcm2target.csv'),
                                         usecols=['tcm_index', 'target_gene_id'])
        self.tcm_target_df.columns = ['drug', 'protein']
        self.dpi_df = pd.concat([self.dpi_df, self.tcm_target_df], axis=0)
        
        self.ppi_df = pd.read_excel(join(self.data_dir, 'ppi.xlsx'),
            usecols=['protein1', 'protein2'])
        self.ppi_df.columns = ['protein_1', 'protein_2']

        self._node_map()

        return self.ddi_df, self.dpi_df, self.ppi_df

    def _node_map(self):
        self.logger.info('Processing data...')
        
        # filter
        protein_list = list(set(self.ppi_df['protein_1']) | set(self.ppi_df['protein_2']))
        protein_list.sort()
        
        self.dpi_df = self.dpi_df[self.dpi_df['protein'].isin(protein_list)]
        drug_list = list(set(self.dpi_df['drug']))
        drug_list.sort()
        self.logger.info(f'{len(drug_list)} drugs have related proteins.')
        
        self.ddi_df = self.ddi_df[(self.ddi_df['drug_1'].isin(drug_list)) &
                                  (self.ddi_df['drug_2'].isin(drug_list))]
        drug_ddi_list = list(set(self.ddi_df['drug_1']) | set(self.ddi_df['drug_2']))
        self.logger.info(f'After filtering, {len(drug_ddi_list)} drugs have DDI.')

        # node map
        all_node_list = drug_list + protein_list
        all_node_map_dict = {n: i for i, n in enumerate(all_node_list)}

        self.ddi_df['drug_1'] = self.ddi_df['drug_1'].map(all_node_map_dict)
        self.ddi_df['drug_2'] = self.ddi_df['drug_2'].map(all_node_map_dict)
        self.ddi_df['label'] = self.ddi_df['label'].apply(lambda x: 1 if x == 'negative' else 0)
        self.logger.info('{} positive DDI, {} negative DDI'.format(len(self.ddi_df[self.ddi_df['label'] == 1]),
                                                                   len(self.ddi_df[self.ddi_df['label'] == 0])))

        self.dpi_df['drug'] = self.dpi_df['drug'].map(all_node_map_dict)
        self.dpi_df['protein'] = self.dpi_df['protein'].map(all_node_map_dict)

        self.ppi_df['protein_1'] = self.ppi_df['protein_1'].map(all_node_map_dict)
        self.ppi_df['protein_2'] = self.ppi_df['protein_2'].map(all_node_map_dict)

        all_node_map_df = pd.DataFrame({'node': list(all_node_map_dict.keys()),
                                        'map': list(all_node_map_dict.values())})
        all_node_map_df.to_csv(join(self.data_dir, 'processed', 'node_map.csv'), index=False)

        self.ddi_df.to_csv(join(self.data_dir, 'processed', 'ddi.csv'), index=False)
        self.dpi_df.to_csv(join(self.data_dir, 'processed', 'dpi.csv'), index=False)
        self.ppi_df.to_csv(join(self.data_dir, 'processed', 'ppi.csv'), index=False)

    def _build_graph(self):
        self.logger.info('Building graph...')
        ppi_edges = self.ppi_df[['protein_1', 'protein_2']].values
        dpi_edges = self.dpi_df[['drug', 'protein']].values
        edges = np.vstack([ppi_edges, dpi_edges])
        tuples = [tuple(x) for x in edges]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def _get_sparse_adj(self):
        def adj_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sparse.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx
        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        self.logger.info('Get sparse adjacency matrix in csr format.')
        # csr matrix, note that if there is a link from node A to B, then the nonzero value in the adjacency matrix is (A, B)
        # where A is the row number and B is the column number
        csr_adjmatrix = nx.adjacency_matrix(self.graph, nodelist=sorted(list(range(0, self.node_num))))

        row_num, col_num = csr_adjmatrix.shape
        self.logger.info('{} edges among {} possible pairs.'.format(csr_adjmatrix.getnnz(), row_num*col_num))

        adj = csr_adjmatrix.tocoo()
        adj = adj_normalize(adj + sparse.eye(row_num))
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        return adj_tensor

    def _create_dataset(self):
        ddi_dataset = ddiDataset(ddi_df=self.ddi_df)
        return ddi_dataset
