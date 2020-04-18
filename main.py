# ### import misc modules
from utils import show_img, load_mnist
import numpy as np

# ### import dataset
from NinaPro import NinaPro

######################################################
# ### import dataset constructors
######################################################
#DONE:
# base: split training and test sets according to number of samples per gesture => num_samples_train = 13050 num_samples_test = 6503
# v1: split training and test sets according to patient numbers num_samples_train => 15615 num_samples_test = 3938
# basevar: same version as base but the window size of a gesture is variable (corrected ground truth using the estimated gesture length) num_samples_train => 2120 num_samples_test = 1070
# v1var: same version as v1 but the window size of a gesture is variable (corrected ground truth using the estimated gesture length) num_samples_train => 2552 num_samples_test = 638

from dataset_constructor.BaselineDataset import BaselineDataset
from dataset_constructor.BaselineVariableWindowDataset import BaselineVariableWindowDataset
from dataset_constructor.LogicalDataset import LogicalDataset
from dataset_constructor.LogicalVariableWindowDataset import LogicalVariableWindowDataset
data_choices = {"base": BaselineDataset, "basevar": BaselineVariableWindowDataset,
                "v1": LogicalDataset, "v1var": LogicalVariableWindowDataset}

######################################################
# ### import feature extractors
######################################################
#DONE:
# rms: 16 features per sample +
# ts: xx features per sample -
# mdwt: 48 features per sample -
# hist: 96 features per sample +
# scat1d: 11200 features per sample - ALLOCATION ERROR
# multirms: 64 features per sample +
# pmrms: 80 features per sample ++ 92.62
# kmrms: 80 features per sample +
# fourier: 96 features per sample +
# none: 200x16 features per sample -

from feature_extractor.AllFeatures import AllFeatures
from feature_extractor.EmptyFeature import EmptyFeature
from feature_extractor.HistogramBins import HistogramBins
from feature_extractor.KMeansRMS import KMeansRMS
from feature_extractor.MarginalDiscreteWaveletTransform import MarginalDiscreteWaveletTransform
from feature_extractor.MultipleRMS import MultipleRMS
from feature_extractor.PaddedMultiRMS import PaddedMultiRMS
from feature_extractor.RMS import RMS
from feature_extractor.ScatteringFeat1D import ScatteringFeat1D
from feature_extractor.TimeStatistics import TimeStatistics
from feature_extractor.FourierBinning import FourierBinning
from feature_extractor.Spectrograms import Spectrograms
feature_choices = {"rms": RMS, "ts": TimeStatistics, "mdwt": MarginalDiscreteWaveletTransform,
                    "hist": HistogramBins, "all": AllFeatures, "scat1d": ScatteringFeat1D,
                    "none": EmptyFeature, "multirms": MultipleRMS, "pmrms": PaddedMultiRMS,
                    "kmrms": KMeansRMS, "fourier": FourierBinning, "specs": Spectrograms}

######################################################
# ### import models and Model/Feature abbreviations and choices.
######################################################

from models.ConvNet import ConvNet
from models.FCNet import FCNet
from models.RandomForest import RandomForest
from models.SVM import SVM
from models.CapsuleNet import CapsuleNet
model_choices = {"rf": RandomForest, "svm": SVM,
                 "fcnet": FCNet, "convnet": ConvNet, "capsnet": CapsuleNet}

def main():
    import argparse
    import os
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="EMG classification (NinaPro5)")
    # model, feature and dataset selectors
    parser.add_argument('--model', default='rf', help=list(model_choices.keys()))
    parser.add_argument('--features', default='rms', help=list(feature_choices.keys()))
    parser.add_argument('--data', default='base', help=list(data_choices.keys()))
    # training params
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--chkpt_period', default=1, type=int)
    parser.add_argument('--valid_period', default=1, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help="Weight decay")
    parser.add_argument('--num_workers', default=1, type=int, help="Number of workers")
    # convolutional network params
    parser.add_argument('--convnet_filters', default=256, type=int)
    parser.add_argument('--convnet_kernel_size', default=9, type=int)
    parser.add_argument('--convnet_stride', default=1, type=int)
    parser.add_argument('--convnet_padding', default=0, type=int)
    parser.add_argument('--convnet_maxpooling', default=3, type=int)
    parser.add_argument('--convnet_fc_num', default=256, type=int)
    # capsule network params
    parser.add_argument('--capsnet_lr_decay', default=0.9, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--capsnet_lam_recon', default=0.0005 * 784, type=float,  help="The coefficient for the loss of decoder")
    parser.add_argument('--capsnet_routings', default=3, type=int, help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--capsnet_conv_filters', default=256, type=int)
    parser.add_argument('--capsnet_conv_kernel_size', default=9, type=int)
    parser.add_argument('--capsnet_conv_stride', default=1, type=int)
    parser.add_argument('--capsnet_conv_padding', default=0, type=int)
    parser.add_argument('--capsnet_primcaps_filters', default=256, type=int)
    parser.add_argument('--capsnet_primcaps_kernel_size', default=5, type=int)
    parser.add_argument('--capsnet_primcaps_stride', default=2, type=int)
    parser.add_argument('--capsnet_primcaps_padding', default=0, type=int)
    parser.add_argument('--capsnet_primcaps_num', default=8, type=int)
    parser.add_argument('--capsnet_digitcaps_dim', default=16, type=int)
    # random forest params
    parser.add_argument('--rf_num_trees', default=32, type=int)
    # support vector machine params
    parser.add_argument('--svm_kernel', default='rbf', type=str, help="linear, poly, rbf, sigmoid or precomputed")
    # directories
    parser.add_argument('--data_dir', default='./', help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--model_dir', default='./saved_models/', help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--model_name', default='', type=str, help="Padded name to the model folder where results and chkpts will be saved")

    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    # feature extractor
    feat_extractor = feature_choices[args.features](data_choices[args.data].__name__)
    # dataset loader
    dataset = data_choices[args.data](os.path.join(args.data_dir, 'NinaPro'), feat_extractor, False)
    if not dataset.load_dataset():
        print("Downloading Ninapro data from the server...")
        data_parser = NinaPro(os.path.join(args.data_dir, 'NinaPro')) #downloader

        print("Loading Ninapro data from processed directory...") #decoder
        loaded_nina = data_parser.load_processed_data()

        print("Extracting dataset features for training, and testing...") #dataset creator (feature extractor)
        dataset.create_dataset(loaded_nina)
    # dataset.train_features 13050x16, dataset.train_labels 13050, dataset.test_features 6503x16, dataset.test_labels 6503
    # model
    classifier = model_choices[args.model](args.model_dir, feat_extractor)

    print("Features summary:")
    print("dataset.train_features:", dataset.train_features.shape, "dataset.train_labels:", dataset.train_labels.shape,
          "dataset.test_features:", dataset.test_features.shape, "dataset.test_labels:", dataset.test_labels.shape)

    #dataset.train_features = dataset.train_features.reshape((dataset.train_features.shape[0], -1))
    #dataset.test_features = dataset.test_features.reshape((dataset.test_features.shape[0], -1))

    train_features = dataset.train_features.astype(float)
    train_labels = dataset.train_labels
    test_features = dataset.test_features.astype(float)
    test_labels = dataset.test_labels
    if len(train_features.shape) == 2:
        train_features = np.expand_dims(train_features, 1)
    if len(test_features.shape) == 2:
        test_features = np.expand_dims(test_features, 1)

    print("Training classifier on training dataset...")
    classifier.train(vars(args), train_features, train_labels, test_features, test_labels, verbose=args.verbose)

    print("Testing classifier on testing dataset...")
    classifier.test(test_features, test_labels)


    #print("Perform inferences...")
    #inferences = classifier.perform_inference(test_features)

    #show_img(x_recon=inferences[1][:3], x_real=test_features[:3], save_dir=os.path.join(classifier.models_path, classifier.__class__.__name__ + "_" + classifier.feat_extractor.__class__.__name__))
    pass

if __name__ == "__main__":
    main()