import os
import argparse
import logging
import time

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, dataset_class, Data_Verifier
from Models.shapeformer import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
import torch

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------- Dataset---------------------------------------------
parser.add_argument("--dataset_pos", default=0, type=int, help="number of processes for extracting shapelets")
parser.add_argument("--num_shapelet", default=3, type=int, help="number of shapelets")
parser.add_argument("--window_size", default=100, type=int, help="window size")

# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['Shapeformer'], choices={'T', 'C-T', 'PPSN', 'Shapeformer'})
# Local Information
parser.add_argument("--len_w", default=64, type=float, help="window size")
parser.add_argument("--local_embed_dim", default=32, type=int, help="embedding dimension of shape")
parser.add_argument("--local_pos_dim", default=32, type=int, help="embedding dimension of pos")

# Global Information
parser.add_argument("--num_pip", default=0.2, type=float, help="number of pips")
parser.add_argument("--sge", default=0, type=int, help="stop-gradient epochs")
parser.add_argument("--shape_embed_dim", default=128, type=int, help="embedding dimension of shape")
parser.add_argument("--pos_embed_dim", default=128, type=int, help="embedding dimension of pos")
parser.add_argument("--processes", default=64, type=int, help="number of processes for extracting shapelets")
parser.add_argument("--pre_shapelet_discovery", default=1, type=int, help="number of processes for extracting shapelets")

# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=64, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=16, help='Number of multi-headed attention heads')
parser.add_argument('--local_num_heads', type=int, default=16, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='Learn',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
parser.add_argument('--dropout', type=float, default=0.4, help='Droupout regularization ratio')

# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='learning rate')
parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1, type=int, help='Seed used for splitting sets')

parser.add_argument("--store_path", default='store/', type=str, help='store path of .pkl files')

args = parser.parse_args()


if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"
    list_dataset_name = os.listdir(config['data_path'])
    list_dataset_name.sort()
    print(list_dataset_name)
    for problem in list_dataset_name[config['dataset_pos']:config['dataset_pos']+1]:  # for loop on the all datasets in "data_dir" directory
        config['data_dir'] = config['data_path'] +"/"+ problem
        print(problem)
        # ------------------------------------ Load Data ---------------------------------------------------------------
        logger.info("Loading Data ...")
        Data = Data_Loader(config)
        train_data = Data['train_data']
        train_label = Data['train_label']
        len_ts = Data['max_len']
        dim = train_data.shape[1]
        print("Number of shapelet %s - %s" % (config['num_shapelet'], config['window_size']))

        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Shapelet Discovery ----------------------------------------------
        shapelet_discovery = ShapeletDiscover(window_size=args.window_size, num_pip=args.num_pip,
                                              processes=args.processes, len_of_ts=len_ts, dim=dim)
        sc_path = os.path.join(config['store_path'], problem + "_" + str(args.window_size) + ".pkl")
        if args.pre_shapelet_discovery == 1:
            shapelet_discovery.load_shapelet_candidates(path=sc_path)
        else:
            time_s = time.time()
            shapelet_discovery.extract_candidate(train_data=train_data)
            shapelet_discovery.discovery(train_data=train_data, train_labels=train_label)
            shapelet_discovery.save_shapelet_candidates(path=sc_path)
            print("shapelet discovery time: %s" % (time.time() - time_s))

        shapelets_info = shapelet_discovery.get_shapelet_info(number_of_shapelet=args.num_shapelet)

        #
        sw = torch.tensor(shapelets_info[:,3])
        sw = torch.softmax(sw*20, dim=0)*sw.shape[0]
        shapelets_info[:,3] = sw.numpy()

        print(shapelets_info.shape)
        shapelets = []
        for si in shapelets_info:
            sc = train_data[int(si[0]), int(si[5]), int(si[1]):int(si[2])]
            shapelets.append(sc)
        config['shapelets_info'] = shapelets_info
        config['shapelets'] = shapelets
        config['len_ts'] = len_ts
        config['ts_dim'] = dim

        train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'])

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Build Model -----------------------------------------------------
        dic_position_results = [config['data_dir'].split('/')[-1]]

        logger.info("Creating model ...")
        config['Data_shape'] = Data['train_data'].shape
        config['num_labels'] = int(max(Data['train_label']))+1

        model = model_factory(config)
        logger.info("Model:\n{}".format(model))
        logger.info("Total number of parameters: {}".format(count_parameters(model)))
        # -------------------------------------------- Model Initialization ------------------------------------
        optim_class = get_optimizer("RAdam")
        config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        config['loss_module'] = get_loss_module()
        save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
        tensorboard_writer = SummaryWriter('summary')
        model.to(device)
        # ---------------------------------------------- Training The Model ------------------------------------
        logger.info('Starting training...')
        trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                    print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
        test_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                          print_interval=config['print_interval'], console=config['console'],
                                          print_conf_mat=False)

        train_runner(config, model, trainer, test_evaluator, save_path)
        best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
        best_model.to(device)

        best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'],
                                                print_interval=config['print_interval'], console=config['console'],
                                                print_conf_mat=True)
        best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
        print_str = 'Best Model Test Summary: '
        for k, v in best_aggr_metrics_test.items():
            print_str += '{}: {} | '.format(k, v)
        print(print_str)
        dic_position_results.append(all_metrics['total_accuracy'])
        problem_df = pd.DataFrame(dic_position_results)
        problem_df.to_csv(os.path.join(config['pred_dir'] + '/' + problem + '.csv'))

        All_Results = np.vstack((All_Results, dic_position_results))

    All_Results_df = pd.DataFrame(All_Results)
    All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))
    print(problem)
