import argparse
import collections
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
from data_loader.data_loaders import ddiDataset
from model.model import DDIGCN as module_arch
from parse_config import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('Inference')

    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    adj = data_loader.get_adj()
    node_num = data_loader.get_node_num()

    logger.info('Loading the dataset...')
    data_inference_df = pd.read_csv('data/tcm/ddi_to_predict.csv')
    node_map_df = pd.read_csv('data/processed/node_map.csv')
    node2id = {row['node']: row['map'] for _, row in node_map_df.iterrows()}
    id2node = {row['map']: row['node'] for _, row in node_map_df.iterrows()}
    data_inference_df['drug1'] = data_inference_df['drug1'].map(node2id)
    data_inference_df['drug2'] = data_inference_df['drug2'].map(node2id)
    data_inference_df['label'] = [-1] * len(data_inference_df)
    
    ddiDataset_inference = ddiDataset(data_inference_df)
    data_loader_inference = torch.utils.data.DataLoader(ddiDataset_inference, batch_size=8, shuffle=False, num_workers=0)

    logger.info('Loading the model...')
    model = module_arch(node_num=node_num,
                        adj=adj,
                        emb_dim=config['arch']['args']['emb_dim'],
                        gcn_layersize=config['arch']['args']['gcn_layersize'],
                        dropout=config['arch']['args']['dropout'])
    resume = 'saved/models/DDI-GCN/1215_152104/model_best.pth'
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    logger.info('Start inference...')
    result_dict = {'drug1': [], 'drug2': [], 'predicted_label': []}
    for i, (drug1, drug2, label) in enumerate(data_loader_inference):
        drug1 = drug1.cuda()
        drug2 = drug2.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = model(drug1, drug2)
            predicted_label = torch.sigmoid(output)
            result_dict['drug1'] += np.squeeze(drug1.cpu().numpy()).tolist()
            result_dict['drug2'] += np.squeeze(drug2.cpu().numpy()).tolist()
            result_dict['predicted_label'] += np.squeeze(predicted_label.cpu().numpy()).tolist()
    result_df = pd.DataFrame(result_dict)
    print(result_df.head())
    result_df['drug1'] = result_df['drug1'].map(id2node)
    result_df['drug2'] = result_df['drug2'].map(id2node)
    result_df.to_csv('data/tcm/ddi_predicted.csv', index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
