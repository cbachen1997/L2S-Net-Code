
import logging
import math
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import copy


import dgl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import skew
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import t
import math


import cv2
from PIL import Image, Image as BICUBIC
import skimage
from skimage.color import label2rgb
from skimage.future import graph
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.feature import graycomatrix, graycoprops
from fast_slic.avx2 import SlicAvx2


import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torchvision import transforms


from tqdm import tqdm


import importlib


from pretrained_model import hrnet
from histocartography.pipeline import PipelineStep


np.seterr(divide='ignore',invalid='ignore')

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"
GPU_DEFAULT_BATCH_SIZE = 16
CHECKPOINT_PATH = "pretrained_model"

COLORMAP = np.array([
    (0, 255, 255),    # Class 1 (urban land)
    (255, 255, 0),    # Class 2 (agriculture land)
    (255, 0, 255),    # Class 3 (rangeland)
    (0, 255, 0),  # Class 4 (forest land)
    (0, 0, 255),  # Class 5 (water)
    (255, 255, 255),   # Class 6 (barren land)
    (0,0,0) # Class 7 (unknown)
])


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)

class MergedSuperpixelExtractor:
    def __init__(self,
        nr_superpixels: Optional[int] = None,
        superpixel_size: Optional[int] = 2500,
        max_nr_superpixels: Optional[int] = None,
        blur_kernel_size: Optional[float] = 1,
        compactness: Optional[int] = 10, # 0.01, 0.1, 1, 10, 100
        max_iterations: Optional[int] = 10,
        threshold: Optional[float] = 0.03,
        connectivity: Optional[int] = 1,
        color_space: Optional[str] = "rgb",
        mergeornot: Optional[bool] = True,
        downsampling_factor: Optional[int] = 1):
        """Extract superpixels with the SLIC algorithm"""
        assert (nr_superpixels is None and superpixel_size is not None) or (
            nr_superpixels is not None and superpixel_size is None
        ), "Provide value for either nr_superpixels or superpixel_size"
        self.nr_superpixels = nr_superpixels
        self.superpixel_size = superpixel_size
        self.max_nr_superpixels = max_nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.compactness = compactness
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.connectivity = connectivity
        self.color_space = color_space
        self.downsampling_factor = downsampling_factor
        self.mergeornot = mergeornot
        
        super().__init__()

    def _get_nr_superpixels(self, image: np.ndarray) -> int:
        """Compute the number of superpixels for initial segmentation
        Args:
            image (np.array): Input tensor
        Returns:
            int: Output number of superpixels
        """
        nr_superpixels = self.nr_superpixels or int(image.shape[0] * image.shape[1] / self.superpixel_size)
        if nr_superpixels is not None and self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_initial_superpixels(self, image: np.ndarray) -> np.ndarray:
        """
        Extract initial superpixels using SLIC
        """

        nr_superpixels = self._get_nr_superpixels(image)


        fast_slic = SlicAvx2(
            num_components=1200, 
            compactness=self.compactness,
            min_size_factor=0.5,
        )


        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        

        superpixels = fast_slic.iterate(image) + 1
        

        regions = regionprops(superpixels)
        

        instance_centroids = np.stack([region.centroid for region in regions])
        instance_centroids = np.round(instance_centroids).astype(int)
        instance_centroids = np.ascontiguousarray(instance_centroids)

        return superpixels, instance_centroids



    def generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph:
        """Construct RAG graph using initial superpixel instance map
        Args:
            input_image (np.ndarray): Input image
            superpixels (np.ndarray): Initial superpixel instance map
        Returns:
            graph: Constructed graph
        """
        g = graph.RAG(superpixels, connectivity=self.connectivity)

        for n in g:
            mask = superpixels == n
            indices = np.nonzero(mask)
            N = len(indices[0])
            g.nodes[n].update(
                {
                    "labels": [n],
                    "N": N,
                    "x": np.sum(input_image[indices], axis=0),
                }
            )
            g.nodes[n]["mean"] = g.nodes[n]["x"] / g.nodes[n]["N"]
            g.nodes[n]["mean"] = g.nodes[n]["mean"] / np.linalg.norm(g.nodes[n]["mean"])

        for x, y, d in g.edges(data=True):
            diff_mean = np.linalg.norm(g.nodes[x]["mean"] - g.nodes[y]["mean"]) / 2
            d["weight"] = diff_mean

        return g 

    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        diff_mean = np.linalg.norm(
            graph.nodes[dst]["mean"] -
            graph.nodes[n]["mean"])

        return {"weight": diff_mean}

    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        graph.nodes[dst]["x"] += graph.nodes[src]["x"]
        graph.nodes[dst]["N"] += graph.nodes[src]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["x"] / \
            graph.nodes[dst]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["mean"] / np.linalg.norm(
            graph.nodes[dst]["mean"]
        )

        """Generate a graph based on the input image and initial superpixel segmentation."""


    def _extract_superpixels(
        self, image: np.ndarray, pre_mask: np.ndarray = None
    ) -> np.ndarray:
        """Perform superpixel extraction
        Args:
            image (np.array): Input tensor
            pre_mask (np.array, optional): Input pre mask
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """

        initial_superpixels, instance_centoirds = self._extract_initial_superpixels(image) 
        if self.mergeornot:
            merged_superpixels = self._merge_superpixels(  
                image, initial_superpixels, pre_mask
            )
            return merged_superpixels, initial_superpixels, instance_centoirds 
        return  None, initial_superpixels, instance_centoirds 

    
    def _downsample(self, image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """
        Downsample an input image with a given downsampling factor.

        Args:
            image (np.array): Input tensor.
            downsampling_factor (int): Factor to downsample.

        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image
    
    def _merge_superpixels(
        self,
        input_image: np.ndarray,
        initial_superpixels: np.ndarray,
        pre_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Merge the initial superpixels to return merged superpixels
        Args:
            image (np.array): Input image
            initial_superpixels (np.array): Initial superpixels
            pre_mask (None, np.array): pre mask
        Returns:
            np.array: Output merged superpixel tensor
        """
        if pre_mask is not None:

            ids_initial = np.unique(initial_superpixels, return_counts=True)
            ids_masked = np.unique(
                pre_mask * initial_superpixels, return_counts=True
            )

            ctr = 1
            superpixels = np.zeros_like(initial_superpixels)
            for i in range(len(ids_initial[0])):
                id = ids_initial[0][i]
                if id in ids_masked[0]:
                    idx = np.where(id == ids_masked[0])[0]
                    ratio = ids_masked[1][idx] / ids_initial[1][i]
                    if ratio >= 0.1:
                        superpixels[initial_superpixels == id] = ctr
                        ctr += 1

            initial_superpixels = superpixels


        g = self.generate_graph(input_image, initial_superpixels)

        merged_superpixels = graph.merge_hierarchical(
            initial_superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merging_function,
            weight_func=self._weighting_function,
        )
        merged_superpixels += 1  
        return merged_superpixels
    
    def process( 
        self, input_image: np.ndarray, pre_mask=None
    ) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            input_image (np.array): Input image.
            pre_mask (None, np.array): pre mask.
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """

        if self.downsampling_factor is not None and self.downsampling_factor != 1:
            input_image = self._downsample(
                input_image, self.downsampling_factor)
            if pre_mask is not None:
                pre_mask = self._downsample(
                    pre_mask, self.downsampling_factor)

        merged_superpixel_map, initial_superpixels, instance_centroids = self._extract_superpixels(
            input_image, pre_mask
        )

        return merged_superpixel_map, initial_superpixels, instance_centroids

def set_graph_on_cuda(graph):
    cuda_graph = dgl.DGLGraph()
    cuda_graph.add_nodes(graph.number_of_nodes())
    cuda_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cuda_graph.ndata[key_graph] = tmp.cuda()
    for key_graph, val_graph in graph.edata.items():
        cuda_graph.edata[key_graph] = graph.edata[key_graph].clone().cuda()
    return cuda_graph


class FeatureExtractor(PipelineStep):
    """Base class for feature extraction"""

    def _process(  
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined instance_map

        Args:
            input_image (np.array): Original RGB image.
            instance_map (np.array): Extracted instance_map.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self._extract_features(input_image, instance_map)

    @abstractmethod
    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """
        Extract features from the input_image for the defined structure.

        Args:
            input_image (np.array): Original RGB image.
            structure (np.array): Structure to extract features.

        Returns:
            torch.Tensor: Extracted features
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """
        Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "features")

    @staticmethod
    def _preprocess_architecture(architecture: str) -> str:
        """
        Preprocess the architecture string to avoid characters that are not allowed as paths.

        Args:
            architecture (str): Unprocessed architecture name.

        Returns:
            str: Architecture name to use for the save path.
        """
        if architecture.startswith("s3://mlflow"):
            processed_architecture = architecture[5:].split("/")
            if len(processed_architecture) == 5:
                _, experiment_id, run_id, _, metric = processed_architecture
                return f"MLflow({experiment_id},{run_id},{metric})"
            elif len(processed_architecture) == 4:
                _, experiment_id, _, name = processed_architecture
                return f"MLflow({experiment_id},{name})"
            else:
                return f"MLflow({','.join(processed_architecture)})"
        elif architecture.endswith(".pth"):

            return architecture

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """
        Downsample an input image with a given downsampling factor.

        Args:
            image (np.array): Input tensor.
            downsampling_factor (int): Factor to downsample.

        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image
    
    
    @staticmethod
    def _upsample(
            image: np.ndarray,
            new_height: int,
            new_width: int) -> np.ndarray:
        """
        Upsample an input image to a speficied new height and width.

        Args:
            image (np.array): Input tensor.
            new_height (int): Target height.
            new_width (int): Target width.

        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image               


class PretrainedSegModel(nn.Module):
    '''
    Using semantic segmentation and morphology methods to get initial high-level entities result
    '''
    def __init__(self, model = 'unet', structure_element_size = 5, min_area = 25, pretrained_model_path = None):
        super(PretrainedSegModel, self).__init__()
        self.structure_element_size = structure_element_size
        self.min_area = min_area
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model == 'unet':
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        elif model == 'hrnet':
            self.model = hrnet.HRnet_Segmentation()
        print('Pretrained model is loaded!')
    def process(self,x):
        
        return self.forward(x)

    def merge_small_node(self, seg_result, min_size=30):
        merge_seg = seg_result.copy()

        labeled_image = skimage.measure.label(seg_result,background=-1)

        region_props = skimage.measure.regionprops(labeled_image)

        region_areas = {region.label: region.area for region in region_props}

        for region in region_props:

            if region.area < min_size:
                label = region.label

                adjacent_labels = set()

                for coord in region.coords:
                    neighbors = [(coord[0] + i, coord[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
                    neighbors = [(i, j) for i, j in neighbors if 0 <= i < labeled_image.shape[0] and 0 <= j < labeled_image.shape[1]]
                    adjacent_labels.update(labeled_image[i, j] for i, j in neighbors)

                adjacent_labels.discard(label)


                max_area_label = max(adjacent_labels, key=lambda label: region_areas[label])


                merge_seg[labeled_image == label] = merge_seg[labeled_image == max_area_label][0]

        labeled_image, num_props = skimage.measure.label(merge_seg, return_num=True,background=-1)

        onehot_label = np.zeros((num_props, 9))
        for i, prop in enumerate(skimage.measure.regionprops(labeled_image)):
            region_label = merge_seg[prop.coords[0][0], prop.coords[0][1]]
            onehot_label[i, region_label-1] = 1

        return labeled_image, onehot_label


    def preprocessing(self,x):
        trf = T.Compose([
        T.ToPILImage(),
        T.ToTensor(), 
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]) 
        inp = trf(x).unsqueeze(0).to('cuda')

        return inp


    def forward(self, x):

        with torch.no_grad():
            seg_mask, feat, seg_image = self.model.detect_image(x,count = False)


        morph_seg, region_label = self.merge_small_node(seg_mask)
        return morph_seg, feat, region_label, seg_image

class PatchFeatureExtractor():
    """Helper class to use a CNN to extract features from an image"""

    def __init__(self, architecture: str, device: torch.device) -> None:
        """
        Create a patch feature extracter of a given architecture and put it on GPU if available.

        Args:
            architecture (str): String of architecture. According to torchvision.models syntax.
            device (torch.device): Torch Device.
        """
        self.device = device

        
        if architecture.startswith("s3://mlflow"):
            model = self._get_mlflow_model(url=architecture)
        elif architecture.endswith(".pth"):
            model = self._get_local_model(path=architecture).to(self.device)
            self.num_features = self._get_num_features(model) # 用RGB_overlap的时候去掉
            self.model = self._remove_classifier(model)
            self.model.eval()
        else:
            model = self._get_torchvision_model(architecture).to(self.device)
            self.num_features = self._get_num_features(model)
            self.model = self._remove_classifier(model)
            self.model.eval()

    @staticmethod
    def _get_num_features(model: nn.Module) -> int:
        """
        Get the number of features of a given model.

        Args:
            model (nn.Module): A PyTorch model.

        Returns:
            int: Number of output features.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            return model.fc.in_features
        else:
            classifier = model.classifier[-1]
            if isinstance(classifier, nn.Sequential):
                classifier = classifier[-1]
            return classifier.in_features
        
    def _get_local_model(self, path: str) -> nn.Module:
        """
        Load a model from a local path.

        Args:
            path (str): Path to the model.

        Returns:
            nn.Module: A PyTorch model.
        """

        state_dict = torch.load(path, map_location=self.device)
        if 'resnet' in path:
            model = torchvision.models.resnet50(pretrained=False,num_classes=30)
        elif 'upernet' in path:
            pass
        else:

            model = hrnet.HRnet(num_classes=8,backbone="hrnetv2_w32",pretrained=False)



        model.load_state_dict(state_dict['model'])

        return model

    def _get_mlflow_model(self, url: str) -> nn.Module:
        """
        Load a MLflow model from a given URL.

        Args:
            url (str): Model url.

        Returns:
            nn.Module: A PyTorch model.
        """
        import mlflow

        model = mlflow.pytorch.load_model(url, map_location=self.device)
        return model

    def _get_torchvision_model(self, architecture: str) -> nn.Module:
        """
        Returns a torchvision model from a given architecture string.

        Args:
            architecture (str): Torchvision model description.

        Returns:
            nn.Module: A pretrained pytorch model.
        """
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class(pretrained=True)
        model = model.to(self.device)
        return model

    @staticmethod
    def _remove_classifier(model: nn.Module) -> nn.Module:
        """
        Returns the model without the classifier to get embeddings.

        Args:
            model (nn.Module): Classifiation model.

        Returns:
            nn.Module: Embedding model.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            model.fc = nn.Sequential()
        else:
            model = model
        return model

    @staticmethod
    def _preprocess_architecture(architecture: str) -> str:
        """
        Preprocess the architecture string to avoid characters that are not allowed as paths.

        Args:
            architecture (str): Unprocessed architecture name.

        Returns:
            str: Architecture name to use for the save path.
        """
        if architecture.startswith("s3://mlflow"):
            processed_architecture = architecture[5:].split("/")
            if len(processed_architecture) == 5:
                _, experiment_id, run_id, _, metric = processed_architecture
                return f"MLflow({experiment_id},{run_id},{metric})"
            elif len(processed_architecture) == 4:
                _, experiment_id, _, name = processed_architecture
                return f"MLflow({experiment_id},{name})"
            else:
                return f"MLflow({','.join(processed_architecture)})"
        elif architecture.endswith(".pth"):
            return architecture

        
    def resize_image(self, image, target_size):

        resize = T.Resize((target_size, target_size), interpolation=BICUBIC)
        image = resize(image)
    
        return image
    

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Computes the embedding of a normalized image input.

        Args:
            image (torch.Tensor): Normalized image input.

        Returns:
            torch.Tensor: Embedding of image.
        """
        if self.model.__class__.__name__ == 'ResNet':
            patch = patch.to(self.device)
            with torch.no_grad():
                embeddings = self.model(patch).squeeze()
            return embeddings
        else:
            patch = np.transpose(patch,(2,0,1))
            patch = torch.from_numpy(patch)
            patch = torch.unsqueeze(patch, dim=0)
            patch = patch.float()
            # patch = torch.transpose(patch,())
            patch = patch.to(self.device)
            with torch.no_grad():
                feature_map = self.model_feat(patch)     

            feature_map = self.resize_image(feature_map, 1024).cpu()
            
            return feature_map

    

class DeepFeatureExtractor(FeatureExtractor):
    """Helper class to extract deep features from instance maps"""

    def __init__(
        self,
        architecture: str,
        patch_size: int,
        resize_size: int = None,
        stride: int = None,
        downsample_factor: int = 1,
        normalizer: Optional[dict] = None,
        batch_size: int = 128,
        fill_value: int = 255,
        num_workers: int = 0,
        verbose: bool = False,
        with_instance_masking: bool = False,
        aggregate_mode: str = 'mean',
        add_class_token = False,
        **kwargs,
    ) -> None:
        """
        Create a deep feature extractor.

        Args:
            architecture (str): Name of the architecture to use. According to torchvision.models syntax.
            patch_size (int): Desired size of patch.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                            patches of size patch_size are provided to the network. Defaults to None.
            stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
            downsample_factor (int): Downsampling factor for image analysis. Defaults to 1.
            normalizer (dict): Dictionary of channel-wise mean and standard deviation for image
                            normalization. If None, using ImageNet normalization factors. Defaults to None.
            batch_size (int): Batch size during processing of patches. Defaults to 32.
            fill_value (int): Constant pixel value for image padding. Defaults to 255.
            num_workers (int): Number of workers in data loader. Defaults to 0.
            verbose (bool): tqdm processing bar. Defaults to False.
            with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
        """
        self.architecture = self._preprocess_architecture(architecture)
        self.patch_size = patch_size
        self.resize_size = resize_size
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride
        self.downsample_factor = downsample_factor
        self.with_instance_masking = with_instance_masking
        self.verbose = verbose

        self.aggregate_mode = aggregate_mode
        self.add_class_token = add_class_token



        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        if normalizer is not None:
            self.normalizer_mean = normalizer.get("mean", [0, 0, 0])
            self.normalizer_std = normalizer.get("std", [1, 1, 1])
        else:

            self.normalizer_mean = [0.485, 0.456, 0.406]
            self.normalizer_std = [0.229, 0.224, 0.225]

        if aggregate_mode == 'CNN':
            self.patch_feature_extractor = PatchFeatureExtractor(
                architecture, device=self.device
            )

        self.fill_value = fill_value
        self.batch_size = batch_size
        self.architecture_unprocessed = architecture
        self.num_workers = num_workers

        if self.num_workers in [0, 1]:
            torch.set_num_threads(1)

    def _collate_patches(self, batch):
        """Patch collate function"""
        instance_indices = [item[0] for item in batch]
        patches = [item[1] for item in batch]
        patches = torch.stack(patches)
        return instance_indices, patches

    def process(
        self, input_image: np.ndarray, instance_map: np.ndarray, instance_label: np.ndarray = None
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined instance_map

        Args:
            input_image (np.array): Original RGB image.
            instance_map (np.array): Extracted instance_map.
            instance_label (np.array): Extracted instance_label. Defaults to None.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self._extract_features(input_image, instance_map, instance_label)

    def _extract_features(
            self,
            input_image: np.ndarray,
            instance_map: np.ndarray,
            instance_label: np.ndarray = None,
            transform: Optional[Callable] = None
        ) -> torch.Tensor:
        """
        Extract features for a given RGB image and its extracted instance_map.
        """

        if input_image.shape[0] != 1024:
            input_image = self._downsample(
                input_image, downsampling_factor=0.25)


        self.properties = regionprops(instance_map)
        vector_length = input_image.shape[-1]
        n_properties = len(self.properties)
        features = None


        def compute_feature(input_image, instance_map, instance_label, vector_length,  features, aggregate_mode):
            if aggregate_mode == 'mean': 

                if input_image.shape[-1] != 3:
                    features = torch.zeros((n_properties, vector_length + 9))
                else:
                    features = torch.zeros((n_properties, vector_length))
                input_image = torch.tensor(input_image, device='cuda')
                instance_map = torch.tensor(instance_map, device='cuda')
                if self.add_class_token:
                    instance_label = torch.tensor(instance_label, device='cuda')
                
                for i, prop in enumerate(self.properties):

                    bbox = prop['bbox']
                    min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
                    label = prop['label']

                    local_label = instance_map[min_x:max_x, min_y:max_y] == label
                    # input_slice = input_image[min_x:max_x, min_y:max_y]
                    input_slice = input_image[min_x:max_x, min_y:max_y].float()
                    superpixel_initial_feature = input_slice * local_label[..., np.newaxis]
                    # superpixel_initial_feature = superpixel_initial_feature.reshape(vector_length, -1)
                    # superpixel_initial_feature = superpixel_initial_feature.view(vector_length, -1)
                    superpixel_initial_feature = superpixel_initial_feature.reshape(vector_length, -1)
                
                    # Handle NaN values in the feature matrix by setting them to zero
                    superpixel_initial_feature[torch.isnan(superpixel_initial_feature)] = 0
                    # Compute the mean of the initial features and assign it to the corresponding row of the features tensor
                    mean_initial_feature = torch.mean(superpixel_initial_feature, dim=1)
                    
                    if self.add_class_token:
                        onehot_label = instance_label[i,:]
                        features[i, :] = torch.cat((mean_initial_feature, onehot_label), dim=0)
                    else:
                        features[i, :] = mean_initial_feature

                return features

                
            elif aggregate_mode == 'PCA':
                for i, prop in tqdm(enumerate(self.properties)):

                    bbox = prop['bbox']
                    min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
                    label = prop['label']

                    local_label = instance_map[min_x:max_x, min_y:max_y] == label
                    input_slice = input_image[min_x:max_x, min_y:max_y]
                    superpixel_initial_feature = input_slice * local_label[..., np.newaxis]
                    superpixel_initial_feature = superpixel_initial_feature.reshape(vector_length, -1)


                    tic = time.time()
                    ipca = IncrementalPCA(n_components=1, batch_size=100)
                    for i in range(0, len(superpixel_initial_feature), 100):
                        batch = superpixel_initial_feature[i:i+100]
                        ipca.partial_fit(batch)

                    features[i, :] = ipca.transform(superpixel_initial_feature).squeeze()
                    toc = time.time()
                    print("PCA took {} seconds".format(toc-tic))

                return features
            
            elif aggregate_mode == 'multi_value':
                centroids = [r.centroid for r in self.properties]
                all_mean_crowdedness, all_std_crowdedness = self._compute_crowdedness(
                    centroids)
                input_image_sq = input_image ** 2
                img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
                node_feat = []
                for region_id, region in enumerate(self.properties):

                    sp_mask = instance_map[region['bbox'][0]:region['bbox'][2], region['bbox'][1]:region['bbox'][3]] == region['label'] 
                    sp_gray = img_gray[region['bbox'][0]:region['bbox'][2], region['bbox'][1]:region['bbox'][3]] * sp_mask
                    
                    area = region["area"]
                    convex_area = region["convex_area"]
                    eccentricity = region["eccentricity"]
                    equivalent_diameter = region["equivalent_diameter"]
                    euler_number = region["euler_number"]
                    extent = region["extent"]
                    filled_area = region["filled_area"]
                    major_axis_length = region["major_axis_length"]
                    minor_axis_length = region["minor_axis_length"]
                    orientation = region["orientation"]
                    perimeter = region["perimeter"]
                    solidity = region["solidity"]
                    convex_hull_perimeter = self._compute_convex_hull_perimeter(sp_mask)
                    roughness = convex_hull_perimeter / perimeter
                    shape_factor = 4 * np.pi * area / convex_hull_perimeter ** 2
                    ellipticity = minor_axis_length / major_axis_length
                    roundness = (4 * np.pi * area) / (perimeter ** 2)

                    feats_shape = [
                        area,
                        convex_area,
                        eccentricity,
                        equivalent_diameter,
                        euler_number,
                        extent,
                        filled_area,
                        major_axis_length,
                        minor_axis_length,
                        orientation,
                        perimeter,
                        solidity,
                        roughness,
                        shape_factor,
                        ellipticity,
                        roundness,
                    ]


                    glcm = graycomatrix(sp_gray, [1], [0])

                    filt_glcm = glcm[1:, 1:, :, :]

                    glcm_contrast = graycoprops(filt_glcm, prop="contrast")
                    glcm_contrast = glcm_contrast[0, 0]
                    glcm_dissimilarity = graycoprops(filt_glcm, prop="dissimilarity")
                    glcm_dissimilarity = glcm_dissimilarity[0, 0]
                    glcm_homogeneity = graycoprops(filt_glcm, prop="homogeneity")
                    glcm_homogeneity = glcm_homogeneity[0, 0]
                    glcm_energy = graycoprops(filt_glcm, prop="energy")
                    glcm_energy = glcm_energy[0, 0]
                    glcm_ASM = graycoprops(filt_glcm, prop="ASM")
                    glcm_ASM = glcm_ASM[0, 0]
                    glcm_dispersion = np.std(filt_glcm)

                    feats_texture = [
                        glcm_contrast,
                        glcm_dissimilarity,
                        glcm_homogeneity,
                        glcm_energy,
                        glcm_ASM,
                        glcm_dispersion,
                    ]

                    feats_crowdedness = [
                        all_mean_crowdedness[region_id],
                        all_std_crowdedness[region_id],
                    ]

                    
                    feats_color = self._color_features_per_channel(input_image, input_image_sq, region['bbox'], region['area'])

                    sp_feats = feats_shape + feats_texture + feats_crowdedness + feats_color
                    features = np.hstack(sp_feats)
                    node_feat.append(features)

                node_feat = np.vstack(node_feat)
                return torch.Tensor(node_feat)
            
            elif aggregate_mode == 'CNN':

                image_dataset = InstanceMapPatchDataset(                        
                    image=input_image,
                    instance_map=instance_map,
                    resize_size=self.resize_size,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    fill_value=self.fill_value,
                    mean=self.normalizer_mean,
                    std=self.normalizer_std,
                    transform=transform,
                    with_instance_masking=self.with_instance_masking,
                )


                image_loader = DataLoader(
                    image_dataset,
                    shuffle=False,
                    batch_size=32,

                    num_workers=self.num_workers,
                    collate_fn=self._collate_patches
                )

                features = torch.empty(
                    size=(
                        len(image_dataset.properties),
                        self.patch_feature_extractor.num_features,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )

   
                embeddings = dict()
                for instance_indices, patches in image_loader:
                    emb = self.patch_feature_extractor(patches)
                    for j, key in enumerate(instance_indices):
                        if key in embeddings:
                            embeddings[key][0] += emb[j]
                            embeddings[key][1] += 1
                        else:
                            embeddings[key] = [emb[j], 1]


                for k, v in embeddings.items():
                    features[k, :] = v[0] / v[1]


                return features.cpu().detach()
            
        features = compute_feature(input_image, instance_map, instance_label, vector_length, features, self.aggregate_mode)
        return features
    
    @staticmethod
    def _color_features_per_channel(
            img_rgb_ch,
            img_rgb_sq_ch,
            mask_idx,
            mask_size):
        codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
        hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
        feats_ = list(hist / mask_size)
        color_mean = np.mean(codes)
        color_std = np.std(codes)
        color_median = np.median(codes)
        color_skewness = skew(codes)

        codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
        color_energy = np.mean(codes)

        feats_.append(color_mean)
        feats_.append(color_std)
        feats_.append(color_median)
        feats_.append(color_skewness)
        feats_.append(color_energy)
        return feats_
    
    @staticmethod
    def _compute_crowdedness(centroids, k=10):
        n_centroids = len(centroids)
        if n_centroids < 3:
            mean_crow = np.array([[0]] * n_centroids)
            std_crow = np.array([[0]] * n_centroids)
            return mean_crow, std_crow
        if n_centroids < k:
            k = n_centroids - 2
        dist = euclidean_distances(centroids, centroids)
        idx = np.argpartition(dist, kth=k + 1, axis=-1)
        x = np.take_along_axis(dist, idx, axis=-1)[:, : k + 1]
        std_crowd = np.reshape(np.std(x, axis=1), newshape=(-1, 1))
        mean_crow = np.reshape(np.mean(x, axis=1), newshape=(-1, 1))
        return mean_crow, std_crowd

    def _compute_convex_hull_perimeter(self, sp_mask):
        """Compute the perimeter of the convex hull induced by the input mask."""
        if cv2.__version__[0] == "3":
            _, contours, _ = cv2.findContours(
                np.uint8(sp_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        elif cv2.__version__[0] == "4":
            contours, _ = cv2.findContours(
                np.uint8(sp_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        hull = cv2.convexHull(contours[0])
        convex_hull_perimeter = cv2.arcLength(hull, True)

        return convex_hull_perimeter



class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
            annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
            add_loc_feats (bool): Flag to include location-based features (ie normalized centroids)
                                in node feature representation.
                                Defaults to False.
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)

    def _process(  
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting pre components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Union[None, np.array], optional): Optional node level to include. Defaults to None.
        Returns:
            dgl.DGLGraph: The constructed graph
        """

        # num_nodes = features.shape[0]
        num_nodes = len(np.unique(instance_map))
        graph = dgl.DGLGraph()
        # graph = dgl.DGLGraphStale()
        graph.add_nodes(num_nodes)


        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        centroids = self._get_node_centroids(instance_map)

        self._set_node_centroids(centroids, graph)

        self._set_node_features(features, image_size, graph)

        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        self._build_topology(instance_map, centroids, graph)
        #!只构造了拓扑关系，没有构造权重
        return graph

    def _get_node_centroids(
            self, instance_map: np.ndarray
    ) -> np.ndarray:
        """Get the centroids of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting pre components
        Returns:
            centroids (np.ndarray): Node centroids
        """
        regions = regionprops(instance_map)
        centroids = np.array([region.centroid for region in regions])
        centroids = np.round(centroids).astype(int)

        return centroids

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)
        # graph.ndata[CENTROID] = torch.tensor(centroids, dtype=torch.float32, device=graph.device)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
                self.add_loc_feats
                and image_size is not None
        ):
            
            centroids = graph.ndata[CENTROID]
            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (
                    features,
                    normalized_centroids
                ),
                dim=concat_dim,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting pre components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting pre components
            centroids (np.array): Node centroids
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "graphs")


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, kernel_size: int = 3, hops: int = 1, **kwargs) -> None:
        """Create a graph builder that uses a provided kernel size to detect connectivity
        Args:
            kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        """
        logging.debug("*** RAG Graph Builder ***")
        assert hops > 0 and isinstance(
            hops, int
        ), f"Invalid hops {hops} ({type(hops)}). Must be integer >= 0"
        self.kernel_size = kernel_size
        self.hops = hops
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation map"""
        assert (
            self.nr_annotation_classes < 256
        ), "Cannot handle that many classes with 8-bits"
        regions = regionprops(instance_map)
        labels = torch.empty(len(regions), dtype=torch.uint8)

        for region_label in np.arange(1, len(regions) + 1):

            histogram = np.zeros(self.nr_annotation_classes, dtype=np.int64) #
            mask = np.ones(len(histogram), np.bool)
            mask[self.annotation_background_class] = 0
            if histogram[mask].sum() == 0:
                assignment = self.annotation_background_class
            else:
                histogram[self.annotation_background_class] = 0
                assignment = np.argmax(histogram)
            labels[region_label - 1] = int(assignment)
        graph.ndata[LABEL] = labels

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Create the graph topology from the instance connectivty in the instance_map"""
        regions = regionprops(instance_map)

        num_instances = len(regions)
        instance_ids = torch.empty(num_instances, dtype=torch.uint8)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        adjacency = np.zeros(shape=(num_instances, num_instances))
        instance_ids = np.arange(1, num_instances+1)
        for instance_id in instance_ids:
            mask = (instance_map == instance_id).astype(np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilation - mask
            idx = pd.unique(instance_map[boundary.astype(bool)])
            instance_id -= 1  
            idx -= 1  
            adjacency[instance_id, idx] = 1

        edge_list = np.nonzero(adjacency)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))


class KNNGraphBuilder(BaseGraphBuilder):
    """
    k-Nearest Neighbors Graph class for graph building.
    """

    def __init__(self, k: int = 5, thresh: int = None, **kwargs) -> None:
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology.

        Args:
            k (int, optional): Number of neighbors. Defaults to 5.
            thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).
        """
        logging.debug("*** kNN Graph Builder ***")
        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation"""
        regions = regionprops(instance_map)
        assert annotation.shape[0] == len(regions), \
            "Number of annotations do not match number of nodes"
        graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Build topology using (thresholded) kNN"""

        adj = kneighbors_graph(
            centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean").toarray()
        if self.thresh is not None:
            adj[adj > self.thresh] = 0


        edge_list = np.nonzero(adj)
        graph.add_edges(edge_list[0].tolist(), edge_list[1].tolist())


class AssignmnentMatrixBuilder(PipelineStep):
    """
    Assigning low-level instances to high-level instances using instance maps.
    """

    def _process(
        self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between low-level and high-level instances
        Args:
            low_level_centroids (np.array): Extracted instance centroids in low-level
            high_level_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Constructed assignment
        """
        return self._build_assignment_matrix(
            low_level_centroids, high_level_map)

    def _build_assignment_matrix(
            self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between inter-level instances"""
        low_level_centroids = low_level_centroids.astype(int)
        high_instance_ids = np.sort(
            pd.unique(np.ravel(high_level_map))).astype(int)
        if 0 in high_instance_ids:
            high_instance_ids = np.delete(
                high_instance_ids, np.where(
                    high_instance_ids == 0))

        low_to_high = high_level_map[
            low_level_centroids[:, 1],
            low_level_centroids[:, 0]
        ].astype(int)

        assignment_matrix = np.zeros( 
            (
                low_level_centroids.shape[0],
                len(high_instance_ids)
            )
        )
        assignment_matrix[np.arange(low_to_high.size), low_to_high - 1] = 1
        return assignment_matrix

def gray2rgb(label_mask):   
    mask=np.zeros((label_mask.shape[0],label_mask.shape[1],3),dtype=np.int8)
    for i in range(6):
        mask[label_mask==i]=COLORMAP[i]
    return mask.astype(np.uint8)


    
class InstanceMapPatchDataset(Dataset):
    """Helper class to use a give image and extracted instance maps as a dataset"""

    def __init__(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        patch_size: int,
        stride: Optional[int],
        resize_size: int = None,
        fill_value: Optional[int] = 255,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        with_instance_masking: Optional[bool] = False,
    ) -> None:
        """
        Create a dataset for a given image and extracted instance map with desired patches
        of (patch_size, patch_size, 3). 

        Args:
            image (np.ndarray): RGB input image.
            instance map (np.ndarray): Extracted instance map.
            patch_size (int): Desired size of patch.
            stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                            patches of size patch_size are provided to the network. Defaults to None.
            fill_value (Optional[int]): Value to fill outside the instance maps. Defaults to 255. 
            mean (list[float], optional): Channel-wise mean for image normalization.
            std (list[float], optional): Channel-wise std for image normalization.
            transform (Callable): Transform to apply. Defaults to None.
            with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
        """
        self.image = image
        self.instance_map = instance_map
        self.patch_size = patch_size
        self.with_instance_masking = with_instance_masking
        self.fill_value = fill_value
        self.stride = stride
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.image = np.pad(
            self.image,
            (
                (self.patch_size, self.patch_size),
                (self.patch_size, self.patch_size),
                (0, 0),
            ),
            mode="constant",
            constant_values=fill_value,
        )
        self.instance_map = np.pad(
            self.instance_map,
            ((self.patch_size, self.patch_size), (self.patch_size, self.patch_size)),
            mode="constant",
            constant_values=0,
        )
        self.patch_size_2 = int(self.patch_size // 2)
        self.threshold = int(self.patch_size * self.patch_size * 0.25)
        self.properties = regionprops(self.instance_map)
        self.warning_threshold = 0.75
        self.patch_coordinates = []
        self.patch_region_count = []
        self.patch_instance_ids = []
        self.patch_overlap = []

        basic_transforms = [transforms.ToPILImage()]# ToPILImage不接受numpy数组，所以要先转换成PIL Image
        
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        if transform is not None:
            basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())

        self.dataset_transform = transforms.Compose(basic_transforms)
        self._precompute()
        self._warning()

    def _add_patch(self, center_x: int, center_y: int, instance_index: int, region_count: int) -> None:
        """
        Extract and include patch information.

        Args:
            center_x (int): Centroid x-coordinate of the patch wrt. the instance map.
            center_y (int): Centroid y-coordinate of the patch wrt. the instance map.
            instance_index (int): Instance index to which the patch belongs.
            region_count (int): Region count indicates the location of the patch wrt. the list of patch coords.
        """
        mask = self.instance_map[
            center_y - self.patch_size_2: center_y + self.patch_size_2,
            center_x - self.patch_size_2: center_x + self.patch_size_2
        ]

        overlap = np.sum(mask == instance_index)
        if overlap > self.threshold:
            loc = [center_x - self.patch_size_2, center_y - self.patch_size_2]
            self.patch_coordinates.append(loc)
            self.patch_region_count.append(region_count)
            self.patch_instance_ids.append(instance_index)
            self.patch_overlap.append(overlap)

    def _get_patch(self, loc: list, region_id: int = None) -> np.ndarray:
        """
        Extract patch from image.

        Args:
            loc (list): Top-left (x,y) coordinate of a patch.
            region_id (int): Index of the region being processed. Defaults to None. 
        """
        min_x = loc[0]
        min_y = loc[1]
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size

        patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])

        if self.with_instance_masking:
            instance_mask = ~(self.instance_map[min_y:max_y, min_x:max_x] == region_id)
            patch[instance_mask, :] = self.fill_value

        return patch

    def _precompute(self):
        """Precompute instance-wise patch information for all instances in the input image."""
        for region_count, region in enumerate(self.properties):

            center_y, center_x = region.centroid
            center_x = int(round(center_x))
            center_y = int(round(center_y))

            min_y, min_x, max_y, max_x = region.bbox


            y_ = copy.deepcopy(center_y) 
            while y_ >= min_y: 
                x_ = copy.deepcopy(center_x) 
                while x_ >= min_x: 
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ -= self.stride


            y_ = copy.deepcopy(center_y)
            while y_ >= min_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ -= self.stride


            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x)
                while x_ >= min_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ += self.stride


            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ += self.stride

    def _warning(self):
        """Check patch coverage statistics to identify if provided patch size includes too much background."""
        self.patch_overlap = np.array(self.patch_overlap) / (
            self.patch_size * self.patch_size
        )
        if np.mean(self.patch_overlap) < self.warning_threshold:
            warnings.warn("Provided patch size is large")
            warnings.warn(
                "Suggestion: Reduce patch size to include relevant context.")
            
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Loads an image for a given patch index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor]: instance_index, image as tensor.
        """
        patch = self._get_patch(
            self.patch_coordinates[index],
            self.patch_instance_ids[index]
        )

        patch = self.dataset_transform(patch)
        
        return self.patch_region_count[index], patch

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.patch_coordinates)
    
def get_stats(array, conf_interval=False, name=None, stdout=False, logout=False):
    """Compute mean and standard deviation from an numerical array
    
    Args:
        array (array like obj): The numerical array, this array can be 
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal. 
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n-1)
        err_bound = t_value * se
    else:
        err_bound = std


    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound

def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


def draw_intermediate_map(image, superpixel_map, segmentation_map, save_path, img_name):

    image = cv2.resize(image, (1024, 1024))

    superpixel_image = label2rgb(superpixel_map, image, kind='avg')
    boundaries = mark_boundaries(superpixel_image, superpixel_map, color=(1,1,0))
    boundaries = (boundaries * 255).astype(np.uint8)

    separator = np.ones((image.shape[0], 10, 3)) * 0 
    combined = np.concatenate((image, separator, boundaries, separator, segmentation_map), axis=1)
    im = Image.fromarray(combined.astype(np.uint8))
    os.makedirs(os.path.join(save_path, 'total_map'), exist_ok=True)
    im.save(os.path.join(save_path, 'total_map', img_name))
