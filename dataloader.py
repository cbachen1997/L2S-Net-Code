"""ESAD Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 


IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'DGLHeteroGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}


def set_graph_on_cuda(graph, device):
    graph = graph.to(device)
    for key, val in graph.ndata.items():
        graph.ndata[key] = val.to(device)
    for key, val in graph.edata.items():
        graph.edata[key] = val.to(device)
    assert graph.device == device, "The graph is not on the specified device."
    return graph


def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class ESAD_Dataset(Dataset):
    """ESAD dataset."""

    def __init__(
            self,
            low_path: str = None,
            high_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = True,
    ):
        """
        xBD dataset constructor.

        Args:
            low_path (str, optional): low_level Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            high_path (str, optional): high_level Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(ESAD_Dataset, self).__init__()

        assert not (low_path is None and high_path is None), "You must provide path to at least 1 modality."

        self.low_path = low_path
        self.high_path = high_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram
        self.path_record = []

        if low_path is not None:
            self._load_cg()

        if high_path is not None:
            self._load_superpixel()

        if assign_mat_path is not None:
            self._load_assign_mat()

    def get_max_graph_node_count(self,cell_graphs):
        max_count = 0
        for graph in cell_graphs:
            max_count = max(max_count, graph.num_nodes())
        return max_count

    def _load_cg(self):
        """
        Load low_level graphs
        """
        self.cg_fnames = glob(os.path.join(self.low_path, '*.bin'))
        self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(fname) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]
            self.max_num_node = self.get_max_graph_node_count(self.cell_graphs)
        
    def _load_superpixel(self):
        """
        Load high_level graphs
        """

        self.sp_fnames = glob(os.path.join(self.high_path, '*.bin'))
        self.sp_fnames.sort()
        self.num_tg = len(self.sp_fnames)
        if self.load_in_ram:
            superpixel_graphs = [load_graphs(fname) for fname in self.sp_fnames]
            self.superpixel_graphs = [entry[0][0] for entry in superpixel_graphs]
            self.superpixel_graphs_labels = [entry[1]['label'].item() for entry in superpixel_graphs]
            self.max_num_node = self.get_max_graph_node_count(self.superpixel_graphs)

    def _load_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(fname).float().t()
                    for fname in self.assign_fnames
            ]
            
    def statistics(self, mod: str = 'low'):
        """
        Get dataset statistics.
        Args:
            mod (str, optional): Modality.'low' or 'high' or 'all' Defaults to 'low'.
        """

        if mod == 'low':
            return self.cell_graphs[0].ndata['feat'].shape[1],\
                max(self.cell_graph_labels) + 1,\
                self.max_num_node
        
        elif mod == 'high':
            return self.superpixel_graphs[0].ndata['feat'].shape[1],\
                max(self.superpixel_graphs_labels) + 1,\
                self.max_num_node
        
        else:
            return self.cell_graphs[0].ndata['feat'].shape[1],\
                max(self.cell_graph_labels) + 1,\
                self.superpixel_graphs[0].ndata['feat'].shape[1],\
                max(self.superpixel_graphs_labels) + 1,\
                self.max_num_node
        
    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                low = self.cell_graphs[index]
                high = self.superpixel_graphs[index]
                assert self.cell_graph_labels[index] == self.superpixel_graphs_labels[index], "The low and high are not the same. There was an issue while creating graph."
                label = self.cell_graph_labels[index]
                assign_mat = self.assign_matrices[index]
            else:
                low, label = load_graphs(self.cg_fnames[index])
                low = low[0]
                label = label['label'].item()
                high, _ = load_graphs(self.sp_fnames[index])
                high = high[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()

            high = dgl.add_self_loop(high)
            low = dgl.add_self_loop(low)

            return low, high, assign_mat, label, self.cg_fnames[index]

        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                high = self.superpixel_graphs[index]
                label = self.superpixel_graphs_labels[index]
            else:
                high, label = load_graphs(self.sp_fnames[index])
                label = label['label'].item()
                high = high[0]
            high = dgl.add_self_loop(high)
            return high, label

        else:
            if self.load_in_ram:
                low = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                low, label = load_graphs(self.cg_fnames[index])
                label = label['label'].item()
                low = low[0]
            low = dgl.add_self_loop(low)
            return low, label

    def __len__(self):
        """Return the number of samples in the xBD dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """

    def collate_fn(batch, id, type):
        """
        Define the collate function to be used based on the modality type.
        :param batch: The input batch of data.
        :param id: The index of the modality in each example.
        :param type: The type of the modality (e.g. "text", "image", "audio").
        :return: The collated batch of data.
        """
        collate_fn = COLLATE_FN[type]

        modality_data = [example[id] for example in batch]

        collated_data = collate_fn(modality_data)
        
        return collated_data


    num_modalities = len(batch[0][:-1]) 

    collated_batch = []
    for mod_id in range(num_modalities):
        mod_type = type(batch[0][mod_id]).__name__
        collated_mod = collate_fn(batch, mod_id, mod_type)
        collated_batch.append(collated_mod)

    labels = []
    for example in batch:
        labels.append(example[-2])

    batched_names = []
    for example in batch:
        batched_names.append(example[-1])

    batched_labels = torch.LongTensor(np.array(labels))

    return collated_batch, batched_labels, batched_names


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a ESAD data loader.
    """

    dataset = ESAD_Dataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader, dataset