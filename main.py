import argparse
import json
import logging
import os
import time
import numpy as np

import dgl
import torch
import torch.nn
import torch.nn.functional as F
from torchsummary import summary
from dgl.data import LegacyTUDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import tqdm
from datetime import date
import random

from dataloader import GraphDataLoader
from networks import SimpleModel, Model
import networkx as nx
from dataloader import make_data_loader
from utils4pre import get_stats
from utils4gnn import MultiScaleHGPSL
from sklearn.metrics import recall_score

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)             
    os.environ['PYTHONHASHSEED'] = str(seed) 
    #! CUDNN对卷积速度的优化，会影响一点精度
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

setup_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="HGP-SL-DGL")
    parser.add_argument("--dataset", type=str, default="Deepglobe",
                        choices=["DD", "PROTEINS", "NCI1", "NCI109", "Mutagenicity", "ENZYMES", "Deepglobe"],
                        help="DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES")
    parser.add_argument(
        '--low_path',
        type=str,
        help='path to the low-level graphs.',
        default='/media/cbachen/Research/4_Experiments/8_SS_SP_integration/results/ESADv2/graphs_low_level',
        # default='/media/cbachen/Research/4_Experiments/3_AD_GCN/Mine/data_stage2_amend/ESAD_V2_low_ResNet_high_Mean_patchsize=32_wo_norm_w_classtoken_for_binary/graphs_low_level',
        required=False
    )
    parser.add_argument(
        '--high_path',
        type=str,
        help='path to high-level graphs.',
        default='/media/cbachen/Research/4_Experiments/8_SS_SP_integration/results/ESADv2/graphs_high_level',
        # default='/media/cbachen/Research/4_Experiments/3_AD_GCN/Mine/data_stage2_amend/ESAD_V2_low_ResNet_high_Mean_patchsize=32_wo_norm_w_classtoken_for_binary/graphs_high_level',
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default='/media/cbachen/Research/4_Experiments/8_SS_SP_integration/results/ESADv2/assignment_matrices',
        # default='/media/cbachen/Research/4_Experiments/3_AD_GCN/Mine/data_stage2_amend/ESAD_V2_low_ResNet_high_Mean_patchsize=32_wo_norm_w_classtoken_for_binary/assignment_matrices',
        required=False
    )
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--train_mode", type=str, default='multi', help="train mode, binary/multi/both")
    parser.add_argument("--sample", type=str, default="true",
                        help="use sample method")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument("--pool_ratio", type=float, default=0.5,
                        help="pooling ratio")
    parser.add_argument("--hid_dim", type =int, default=128,
                        help="hidden size")
    parser.add_argument("--conv_layers", type=int, default=3,
                        help="number of conv layers")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout ratio")
    parser.add_argument("--lamb", type=float, default=1.0,
                        help="trade-off parameter")
    parser.add_argument("--epochs", type=int, default=300,
                        help="max number of training epochs")
    parser.add_argument("--patience", type=int, default=50,
                        help="patience for early stopping")
    parser.add_argument("--device", type=int, default=0,
                        help="device id, -1 for cpu")
    parser.add_argument("--dataset_path", type=str, default="./dataset",
                        help="path to dataset")
    parser.add_argument("--print_every", type=int, default=10,
                        help="print trainlog every k epochs, -1 for silent training")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="number of trials")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers")
    parser.add_argument("--in_ram", type=bool, default=True, help='Whether load dataset in ram')
    parser.add_argument("--output_path", type=str, default="./HGP-SL-dgl/output/")
    parser.add_argument("--mod", type=str, default="all", help='low or high or all')
    parser.add_argument("--layer_type", type=str, default='GATv2', help='WConv / GAT / GATv2 / GIN / SAGE / PNA / EConv / MRConv')
    parser.add_argument("--agg_mode", type=str, default='LSTM', help='sum/mean/concat/LSTM/None')
    parser.add_argument("--ass", type=bool, default=True, help='weather use AM')
    parser.add_argument("--fusion_mode", type=str,default='AFM', help='sum/mean/concat/AFM')
    
    args = parser.parse_args()


    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    if args.print_every == -1:
        args.print_every = args.epochs + 1

    if args.sample.lower() == "true":
        args.sample = True
    else:
        args.sample = False

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)  
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path
                    )
    today = date.today().strftime("%Y-%m-%d")
    name = "HGP-SL-{}-PRE-{}_Data={}_Layers={}_Epochs={}_Pool={}_WeightDecay={}_Lr={}_Mode={}_FusionMode={}_w_AM_patchsize_32_RSPMean_OEM".format(
        args.layer_type, args.agg_mode, args.dataset, args.conv_layers, args.epochs, args.pool_ratio, args.weight_decay,\
        args.lr, args.mod, args.fusion_mode)
    name = "{}_{}".format(today, name)  
    args.output_path = os.path.join(args.output_path, name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


def calculate_average_graph(data_loader):
    total_low_nodes = 0
    total_high_nodes = 0
    total_low_edges = 0
    total_high_edges = 0
    num_samples = len(data_loader.dataset)
    len_loader = len(data_loader)

    progress_bar = tqdm.tqdm(total=len_loader, desc='Calculating average graph')

    for batch in data_loader:
        batch_graphs = batch[0]
        # batch_high_graphs = batch[1]
        batch_low_graphs = batch_graphs[0]
        batch_high_graphs = batch_graphs[1]
        total_low_nodes += batch_low_graphs.number_of_nodes()
        total_low_edges += batch_low_graphs.number_of_edges()

        # for high_graph in batch_high_graphs:
        total_high_nodes += batch_high_graphs.number_of_nodes()
        total_high_edges += batch_high_graphs.number_of_edges()

        progress_bar.update(1)
    
    progress_bar.close()

    average_low_nodes = total_low_nodes // num_samples
    average_high_nodes = total_high_nodes // num_samples
    average_low_edges = total_low_edges // num_samples
    average_high_edges = total_high_edges // num_samples

    print("Average number of low-level nodes: {}, edge: {}".format(average_low_nodes, average_low_edges))
    print("Average number of high-level nodes: {}, edge: {}".format(average_high_nodes, average_high_edges))

    return average_low_nodes, average_low_edges, average_high_nodes, average_high_edges

def train(args, model:torch.nn.Module, optimizer, trainloader, device, epoch):
    model.train()
    total_loss = 0.
    total_recall = 0.00001
    correct = 0.
    num_batches = len(trainloader)
    num_samples = len(trainloader.dataset)
    loss_fn_bin = torch.nn.BCELoss()
    loss_fn_multi = torch.nn.CrossEntropyLoss()
    for batch in tqdm.tqdm(trainloader, position=1, desc="Training"):
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch[0], batch[1]
        if len(batch_graphs) == 1:
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)
            out = model(batch_graphs, batch_graphs.ndata["feat"])
        else:
            out = model(batch_graphs[0].to(device), batch_graphs[1].to(device), batch_graphs[2])

            binary_outputs = out[0]  
            multi_outputs = out 
            batch_labels = batch_labels.to(device)
        if args.train_mode == 'binary':
            batch_labels_oh = F.one_hot(batch_labels, num_classes=2).to(device)
            loss = loss_fn_bin(out, batch_labels_oh.float())
        elif args.train_mode == 'multi':
            loss = loss_fn_multi(out, batch_labels)
        elif args.train_mode == 'both':
            # 拆分 batch_labels
            binary_labels = batch_labels[:, 0]  # 二分类标签
            binary_labels = F.one_hot(binary_labels, num_classes=2).to(device)
            loss1 = loss_fn_bin(binary_outputs, binary_labels.float())

            multi_labels = batch_labels[:, 1]   # 多分类标签
            # multi_labels = F.one_hot(multi_labels, num_classes=13).to(device)
            loss2 = loss_fn_multi(multi_outputs, multi_labels)
            # 根据 batch_labels[0] 的值组合损失
            loss = loss1 + loss2
        

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct += pred.eq(batch_labels).sum().item()
            total_recall += recall_score(batch_labels.cpu(), pred.cpu(), average='macro', zero_division=1)

    train_acc = correct / num_samples
    train_loss = total_loss / num_batches
    train_recall = total_recall / num_batches
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Train Recall: {train_recall:.3f}, LR: {optimizer.param_groups[0]['lr']}")
    # logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Train Recall: {train_recall:.3f}, LR: {optimizer.param_groups[0]['lr']}")
    
    return train_loss, train_acc, train_recall

def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.
    total_loss = 0.
    total_recall = 0.
    num_batches = len(loader)
    num_graphs = len(loader.dataset)
    loss_fn = torch.nn.BCELoss()

    event_start = torch.cuda.streams.Event(enable_timing=True)
    event_end = torch.cuda.streams.Event(enable_timing=True)
    infer_time = []

    for batch in tqdm.tqdm(loader, position=1, desc='Testing'):
        batch_graphs, batch_labels = batch[0], batch[1]
        if len(batch_graphs) == 1:
            batch_graphs = batch_graphs[0].to(device)
            batch_labels = batch_labels.to(device)
            
            # out = model(batch_graphs, batch_graphs.ndata["feat"])
            out = model(batch_graphs[0].to(device), batch_graphs[1].to(device), batch_graphs[2])
            # print("Time: ", toc - tic)
        else:
            batch_labels = batch_labels.to(device)
            batch_labels_oh = F.one_hot(batch_labels, num_classes=2).to(device)
            event_start.record()
            # out = model(batch_graphs[0].to(device), batch_graphs[1].to(device))
            out = model(batch_graphs[0].to(device), batch_graphs[1].to(device), batch_graphs[2])
            event_end.record()
            # print("Time: ", toc - tic)
        time.sleep(0.1)
        inference_time_ms = event_start.elapsed_time(event_end)
        infer_time.append(inference_time_ms)
        loss = loss_fn(out, batch_labels_oh.float())
        total_loss += loss.item()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct += pred.eq(batch_labels).sum().item()
            total_recall += recall_score(batch_labels.cpu(), pred.cpu(), average='macro', zero_division=1)
    accuracy = correct / num_graphs
    avg_loss = total_loss / num_batches
    avg_recall = total_recall / num_batches
    avg_inf_time = sum(infer_time)/(len(loader)*args.batch_size)
    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.3f}, Test Recall: {avg_recall:.3f}")
    print("Inference time per image (ms):", avg_inf_time)
    return accuracy, avg_loss, avg_recall, avg_inf_time

import csv

@torch.no_grad()
def test_filter(model: torch.nn.Module, loader, device):
    model.eval()
    FP_images, TP_images, FN_images, TN_images = [], [], [], []
    for batch in tqdm.tqdm(loader, position=1, desc='Testing'):
        low = batch[0][0].to(device)
        high = batch[0][1].to(device)
        ass = batch[0][2]
        batch_labels = batch[1].to(device)
        image_names = batch[2]
        out = model(low,high,ass)
        pred = out.argmax(dim=1)
        for i in range(len(batch_labels)):
            if batch_labels[i] == 1 and pred[i] == 1:
                TP_images.append((image_names[i], out[i][1].item()))
            elif batch_labels[i] == 0 and pred[i] == 1:
                FP_images.append((image_names[i], out[i][1].item()))
            elif batch_labels[i] == 1 and pred[i] == 0:
                FN_images.append((image_names[i], out[i][0].item()))
            elif batch_labels[i] == 0 and pred[i] == 0:
                TN_images.append((image_names[i], out[i][0].item()))
    TP_images = sorted(TP_images, key=lambda x: x[1], reverse=True)
    FP_images = sorted(FP_images, key=lambda x: x[1], reverse=True)
    FN_images = sorted(FN_images, key=lambda x: x[1], reverse=True)
    TN_images = sorted(TN_images, key=lambda x: x[1], reverse=True)
    with open('TP_images.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, item in enumerate(TP_images, 1):
            writer.writerow([f"{i}_{item[0]}", item[1]])
    with open('FP_images.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, item in enumerate(FP_images, 1):
            writer.writerow([f"{i}_{item[0]}", item[1]])
    with open('FN_images.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, item in enumerate(FN_images, 1):
            writer.writerow([f"{i}_{item[0]}", item[1]])
    with open('TN_images.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, item in enumerate(TN_images, 1):
            writer.writerow([f"{i}_{item[0]}", item[1]])

@torch.no_grad()
def val(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.
    total_loss = 0.
    total_recall = 0.
    num_batches = len(loader)
    num_graphs = len(loader.dataset)
    loss_fn_bin = torch.nn.BCELoss()
    loss_fn_multi = torch.nn.CrossEntropyLoss()
    for batch in tqdm.tqdm(loader, position=1, desc='Validating'):
        batch_graphs, batch_labels = batch[0], batch[1]
        if len(batch_graphs) == 1:
            batch_graphs = batch_graphs[0].to(device)
            batch_labels = batch_labels.to(device)
        else:
            out = model(batch_graphs[0].to(device), batch_graphs[1].to(device), batch_graphs[2])
            binary_outputs = out[0]  
            multi_outputs = out  
            batch_labels = batch_labels.to(device)

        #损失计算
        if args.train_mode == 'binary':
            batch_labels_oh = F.one_hot(batch_labels, num_classes=2).to(device)
            loss = loss_fn_bin(out, batch_labels_oh.float())
        elif args.train_mode == 'multi':
            loss = loss_fn_multi(out, batch_labels)
            
        total_loss += loss.item()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct += pred.eq(batch_labels).sum().item()
            total_recall += recall_score(batch_labels.cpu(), pred.cpu(), average='macro', zero_division=1)
    accuracy = correct / num_graphs
    avg_loss = total_loss / num_batches
    avg_recall = total_recall / num_batches
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.3f}, Val Recall: {avg_recall:.3f}")
    return accuracy, avg_loss, avg_recall




def main(args, num_trials):
    """
    This function prepares the graph data and retrieves train/validation/test index.
    It creates model, training components and trains the model for given number of epochs.
    Finally, it returns the final test accuracy and average time taken per epoch for training.

    Args:
    - args: command line arguments
    - num_trials: number of trials to run
    Returns:
    - final_test_acc: final test accuracy after training
    - avg_train_time_per_epoch: average time taken per epoch for training
    """

    train_loader, train_dataset = make_data_loader(
        low_path=os.path.join(args.low_path, 'train') if args.low_path is not None else None,
        high_path=os.path.join(args.high_path, 'train') if args.high_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram, 
        train_mode=args.train_mode,
        num_workers=args.num_workers
    )
    val_loader, _ = make_data_loader(
        low_path=os.path.join(args.low_path, 'val') if args.low_path is not None else None,
        high_path=os.path.join(args.high_path, 'val') if args.high_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'val') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram, 
        train_mode=args.train_mode,
        num_workers=args.num_workers
    )
    test_loader, _ = make_data_loader(
        low_path=os.path.join(args.low_path, 'test') if args.low_path is not None else None,
        high_path=os.path.join(args.high_path, 'test') if args.high_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram, 
        train_mode=args.train_mode,
        num_workers=args.num_workers
    )



    device = torch.device(args.device)
    

    model = MultiScaleHGPSL(num_layers = args.conv_layers,\
                            dropout_rate = args.dropout, pool_ratio = args.pool_ratio, lamb = args.lamb,sample=args.sample,\
                            layer_type=args.layer_type, agg_mode=args.agg_mode, ass=args.ass, mod=args.mod, fusion_mode = args.fusion_mode, train_mode=args.train_mode ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.output_path + '/output.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    bad_cound = 0
    best_val_loss = 10000.0
    best_val_acc = 0.
    best_val_recall = 0.
    final_test_acc = 0.
    best_epoch = 0
    train_times = []
    train_losses = []
    train_accs = []
    train_recalls = []
    val_losses = []
    val_accs = []
    val_recalls = []
    inf_time_list = []
    for e in tqdm.tqdm(range(args.epochs), position=0, desc="Training progress"):
        s_time = time.time()

        train_loss, train_acc, train_recall = train(args, model, optimizer, train_loader, device, epoch=e)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_recalls.append(train_recall)
        train_times.append(time.time() - s_time)

        if (e + 1) % args.print_every == 0:
            val_acc, val_loss, val_recall = val(model, val_loader, device)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            val_recalls.append(val_recall)
            logger.info(f"Epoch {e+1}: train_loss={train_loss}, train_acc={train_acc}, train_recall={train_recall}, val_loss={val_loss}, val_acc={val_acc}, val_recall={val_recall}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                final_test_acc = val_acc
                best_epoch = e
                bad_cound = 0
                best_loss_model_state_dict = model.state_dict()
                torch.save(best_loss_model_state_dict, os.path.join(args.output_path, f"best_val_loss_model.pt"))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_model_state_dict = model.state_dict()
                torch.save(best_acc_model_state_dict, os.path.join(args.output_path, f"best_val_acc_model.pt"))
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_recall_model_state_dict = model.state_dict()
                torch.save(best_recall_model_state_dict, os.path.join(args.output_path, f"best_val_recall_model.pt"))
            else:
                bad_cound += 1
            if bad_cound == args.patience:
                break
            scheduler.step(val_loss) 
        else:
            logger.info(f"Epoch {e+1}: train_loss={train_loss}, train_acc={train_acc}, train_recall={train_recall}")
        

    logger.removeHandler(handler)
    handler.close()
    logger = None
    print("Best Epoch {}, final test acc {:.4f}".format(best_epoch, final_test_acc))

    avg_train_time_per_epoch = sum(train_times) / len(train_times)
    return best_val_acc, avg_train_time_per_epoch





if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    avg_inf_time_list = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, train_time = main(args, num_trials=i)
        res.append(acc)
        train_times.append(train_time)

    mean, err_bd = get_stats(res, conf_interval=False)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))
    out_dict = {"hyper-parameters": vars(args),
                "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
                "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
                }
    
    with open(args.output_path + '/out_dict.json', "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
