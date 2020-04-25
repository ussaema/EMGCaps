from abc import ABC, abstractmethod
import torch
import torchnet
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

try:
    import cPickle as pickle
except:
    import pickle

# For all torch models
torch.manual_seed(13)

NUM_GESTURES = 53 # 52 + 1 (rest)


class ClassifierModel(ABC):
    #
    # All classifier models perform this basic functionality
    #

    def __init__(self, models_path, feat_extractor):
        """
        :param models_path: Path to store model
        :param feat_extractor: FeatureExtractor object, turning raw EMG/IMU samples into a feature vector
        """
        self.models_path    = models_path
        self.feat_extractor = feat_extractor
        self.num_samples    = None

    @abstractmethod
    def train(self, args, train_features, train_labels, valid_features = None, valid_labels = None):
        """
        :param train_features: A numpy array, a member of a Dataset object, containing features for training
        :param train_labels:  A numpy array, a member of a Dataset object, containing label for training
        :param valid_features: A numpy array, a member of a Dataset object, containing features for validation
        :param valid_labels:  A numpy array, a member of a Dataset object, containing label for validation
        """

        params = {}
        for key, val in args.items():
            if args['model'] in key:
                params[key] = val
        self.args = params
        self.__dict__.update(params)
        model_name = self.model_name = args['model_name']
        if len(self.model_name):
            model_name = "_" + model_name
        self.full_model_path = os.path.join(self.models_path, self.__class__.__name__ + model_name + "_" + self.feat_extractor.dataset_name + "_" + self.feat_extractor.__class__.__name__)
        if not os.path.exists(self.full_model_path):
            os.makedirs(self.full_model_path)
        pass
        #self.save_log(self.full_model_path)

    @abstractmethod
    def test(self, test_features, test_labels):
        """
        :param test_features: A numpy array, a member of a Dataset object, containing features for testing
        :param test_labels:  A numpy array, a member of a Dataset object, containing label for testing
        """
        pass

    def perform_inference(self, test_features):
        """
            Given test features and labels, compute predictions and classifier accuracy,

        :param test_features:Features from the test split.
        :param test_labels: Labels from the test split.
        :return: * (test_labels != None) Classifier accuracy from 0 ~ 1.0.
                 * (test_labels == None) Predictions
        """
        predictions = self.perform_inference_helper(test_features)

        return predictions


    @abstractmethod
    def perform_inference_helper(self, test_features):
        """
            Helper function for above
        """
        pass

    @abstractmethod
    def save_figure(self, path):
        pass

    '''def save_model(self, dir_path = None):
        """
            Serializes (this) object for future loading and use

        :param path: Directory path to save this object
        """
        model_feat_name = self.feat_extractor.dataset_name + "_" + self.__class__.__name__ + "_" + self.feat_extractor.__class__.__name__

        if dir_path is None:
            dir_path = os.path.join(self.models_path, model_feat_name)
        else:
            dir_path = os.path.join(dir_path, model_feat_name)

        with open(dir_path, 'wb') as f:
            pickle.dump(self, f, 2)'''

    '''def load_model(path):
        """
            De-serializes a pickled (serialized) ClassifierModel object

        :param: Path of pickled (serialized) object

        :return A ClassifierModel object
        """
        if not (os.path.exists(path)):
            return

        with open(path, 'rb') as f:
            object = pickle.load(f)
            return object'''

    def classifier_accuracy(self, predictions, test_labels):
        """
        :param predictions: Predictions made on a set of test_features
        :param test_labels: Ground truth labels associated with test features

        :return: Classifier accuracy from 0 ~ 1.0.
        """
        errors      = predictions == test_labels
        acc_rate    = len([x for x in errors if (x == True)]) / len(errors)
        return acc_rate

    def per_class_accuracy(self, test_features, test_labels):
        """
        :param test_features: Features used to generate predictions
        :param test_labels: Ground truth labels associated with test features

        :return: [dict] : Each key refers to a per class accuracy
        """

        pred_labels = self.perform_inference_helper(test_features)

        test_labels_found = {}
        for x in test_labels:
            if x not in test_labels_found:
                test_labels_found[x] = []

        for i in range(test_labels.shape[0]):
            cur_label   = test_labels[i]
            cur_pred    = pred_labels[i]
            test_labels_found[cur_label].append(cur_label == cur_pred)

        for y in test_labels_found.keys():
            sum = 0
            for j in test_labels_found[y]:
                sum += j

            test_labels_found[y] = sum / len(test_labels_found[y])

        return test_labels_found


#
# All Torch classifiers implement this interface.
#
class TorchModel(ClassifierModel):
    # Configurable
    checkpoint_ext   = ".chkpt"
    needs_train_feat = False

    # Do not modify (states)
    start_epoch     = 0
    train_accs      = []
    test_accs       = []
    train_losses = []
    test_losses = []
    valid_features  = None
    valid_labels    = None

    def __init__(self, models_path, feat_extractor):
        super().__init__(models_path, feat_extractor)
        # Engine for training/testing model
        self.hooks = {}
        self.hooks['on_start'] = self.on_start
        self.hooks['on_start_epoch'] = self.on_start_epoch
        self.hooks['on_sample'] = self.on_sample # before optimization step
        self.hooks['on_forward'] = self.on_forward # after optimization step
        self.hooks['on_end_epoch']   = self.on_end_epoch
        self.hooks['on_end']         = self.on_end

        self.meter_loss = torchnet.meter.AverageValueMeter()
        self.classerr   = torchnet.meter.ClassErrorMeter(accuracy=True)

        # Note: Use GPU by default
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model  = None

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def perform_inference(self, test_features):
        """
            Given test features and labels, compute predictions and classifier accuracy,

        :param test_features:Features from the test split.
        :param test_labels: Labels from the test split.
        :return: * (test_labels != None) Classifier accuracy from 0 ~ 1.0.
                 * (test_labels == None) Predictions
        """
        self.model.eval()

        outputs = []
        for i in range(len(test_features) // self.batch_size):
            outputs.append(self.perform_inference_helper(test_features[i:i+self.batch_size]))
        if len(test_features) % self.batch_size != 0:
            outputs.append(self.perform_inference_helper(test_features[(len(test_features) // self.batch_size)*self.batch_size:len(test_features)]))
        predictions = []
        for i in range(len(outputs[0])):
            predictions.append(np.concatenate([out[i] for out in outputs]))
        return predictions

    def test(self, test_features, test_labels):
        """
        :param test_features: A numpy array, a member of a Dataset object, containing features for testing
        :param test_labels:  A numpy array, a member of a Dataset object, containing label for testing
        """
        # Use inference model
        self.model.eval()
        self.reset_meters()

        torch_dataset = torchnet.dataset.TensorDataset([
            (torch.from_numpy(test_features)).float(),
            torch.from_numpy(test_labels)
        ])
        iterator = torch_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

        state = {
            'network': self.forward_pass,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        test_acc = self.classerr.value()
        test_loss = self.meter_loss.value()
        try:
            self.epochs_iterator.set_description('Testing accuracy: %.2f, Testing loss: %.2f' % (test_acc[0], test_loss[0]))
        except:
            pass
        self.hook('on_end', state)
        return test_acc, test_loss

    def train(self, args, train_features, train_labels, valid_features = None, valid_labels = None, verbose=1):
        """
        :param train_features: A numpy array, a member of a Dataset object, containing features for training
        :param train_labels:  A numpy array, a member of a Dataset object, containing label for training
        :param valid_features: A numpy array, a member of a Dataset object, containing features for validation
        :param valid_labels:  A numpy array, a member of a Dataset object, containing label for validation
        """
        super().train(args, train_features, train_labels, valid_features, valid_labels)
        self.epochs = args['epochs']
        self.chkpt_period = args['chkpt_period']
        self.valid_period = args['valid_period']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.train_features = train_features
        self.train_labels = train_labels
        self.valid_features = valid_features
        self.valid_labels   = valid_labels
        self.verbose = verbose

        dim_in = train_features[0].shape

        if self.needs_train_feat:
            self.model  = self.define_model(dim_in, train_features)
        else:
            self.model  = self.define_model(dim_in)

        # Use model for training
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.load_checkpoint(self.full_model_path)
        self.save_log(self.full_model_path)
        if self.start_epoch < self.epochs:

            torch_dataset = torchnet.dataset.TensorDataset([
                (torch.from_numpy(train_features)).float(),
                torch.from_numpy(train_labels)
            ])
            iterator = torch_dataset.parallel(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )

            #self.engine.train(self.forward_pass, iterator, maxepoch=self.epochs - self.start_epoch, optimizer=self.optimizer)
            state = {
                'network': self.forward_pass,
                'iterator': iterator,
                'maxepoch': self.epochs - self.start_epoch,
                'optimizer': self.optimizer,
                'epoch': 0,
                't': 0,
                'train': True,
            }

            self.hook('on_start', state)
            if verbose:
                self.epochs_iterator = tqdm(total=self.epochs, initial=self.start_epoch, position=0)
            while state['epoch'] < state['maxepoch']:
                self.hook('on_start_epoch', state)
                if verbose:
                    self.batches_iterator = tqdm(total = len(state['iterator']), position=1)
                for sample in state['iterator']:
                    #sample[0] = sample[0].permute(0, 3, 1, 2).float()
                    #print(sample[0].max(), sample[0].min(), sample[0].shape, sample[0].cuda, train_features.max(), train_features.min(), train_features.shape)
                    state['sample'] = sample
                    self.hook('on_sample', state)
                    def closure():
                        loss, output = state['network'](state['sample'])
                        state['output'] = output
                        state['loss'] = loss
                        loss.backward()
                        self.hook('on_forward', state)
                        if verbose:
                            self.batches_iterator.set_description('Training accuracy: %.2f, Training loss: %.2f' % (self.classerr.value()[0], self.meter_loss.value()[0]))

                        # to free memory in save_for_backward
                        state['output'] = None
                        state['loss'] = None
                        return loss

                    state['optimizer'].zero_grad()
                    state['optimizer'].step(closure)
                    state['t'] += 1
                    if self.batches_iterator.n < self.batches_iterator.total and verbose:
                        self.batches_iterator.update(1)
                self.hook('on_end_epoch', state)
                state['epoch'] += 1

                if verbose:
                    self.epochs_iterator.update(1)
            self.hook('on_end', state)
            return state

        print("Training complete ({} epochs)...".format(self.epochs))


    def perform_inference_helper(self, test_features):
        """
            Helper function for ClassifierModel's perform_inference(self, test_features, test_labels)
        """

        # Use inference model
        self.model.eval()

        outputs = (self.model((torch.from_numpy(test_features).float()).to(self.device)))
        predictions = outputs[0].cpu().detach() if outputs.__class__.__name__ in ('list', 'tuple') else outputs.cpu().detach()
        predictions = (torch.argmax(predictions, 1)).numpy()

        return predictions, np.array([out.cpu().detach().numpy() for out in outputs[1:]]).squeeze(axis=0) if outputs.__class__.__name__ in ('list', 'tuple') else None

    def reset_meters(self):
        """
            Resets meters used for computing classification accuracy
        """
        self.classerr.reset()
        self.meter_loss.reset()

    def on_sample(self, state):
        """
            For each batch of training/testing samples in the training/testing split, this function is called to
                determine if the sample is for training/testing (by other functions used by Torch engine)

        :param state: A state object used by the Torch engine
        """
        state['sample'].append(state['train'])


    def on_forward(self, state):
        """
            For each batch of training/testing samples in the training/testing split, this function is called to
                append classification accuracy and loss

        :param state: A state object used by the Torch engine
        """
        self.classerr.add(state['output'][0].data, torch.LongTensor(state['sample'][1].type(torch.LongTensor)))
        self.meter_loss.add(state['loss'].item())

    def on_start(self, state):
        """
            On the start of a training/testing epoch, this function is called, to reset meters (used for class. acc.)

        :param state: A state object used by the Torch engine
        """
        pass

    def on_start_epoch(self, state):
        """
            On the start of a training/testing epoch, this function is called, to reset meters (used for class. acc.)

        :param state: A state object used by the Torch engine
        """
        self.reset_meters()

    def on_end_epoch(self, state):
        """
            On the end of a training/testing epoch, this function is called, to save a checkpoint and append
                training/testing accuracy

        :param state: A state object used by the Torch engine
        """
        #print('Training Accuracy (epoch', int(state["epoch"]) + self.start_epoch, '): ', self.classerr.value())
        self.train_accs.append(self.classerr.value()[0])
        self.train_losses.append(self.meter_loss.value()[0])

        if ((int(int(state["epoch"]) + self.start_epoch) % self.valid_period == 0) and
                ((self.valid_labels is not None) and (self.valid_features is not None)) and
                (state["epoch"] != self.epochs)):
            self.reset_meters()
            self.test(self.valid_features, self.valid_labels)
            self.model.train()

        if int(int(state["epoch"]) + self.start_epoch) % self.chkpt_period == 0:
            self.save_checkpoint(int(state["epoch"]) + 1 + self.start_epoch, state["loss"], self.full_model_path)
            self.save_figure(self.full_model_path)
            self.save_log(self.full_model_path)

    def on_end(self, state):
        """
            On the end of training/testing, this function is called, to append training/test accuracy

        :param state: A state object used by the Torch engine
        """
        #print('Training' if state['train'] else 'Testing', 'accuracy')

        if (not state["train"]) and (self.valid_features is not None) and (self.valid_labels is not None):
            #print(self.classerr.value())
            self.test_accs.append(self.classerr.value()[0])
            self.test_losses.append(self.meter_loss.value()[0])

        elif state["train"]:
            pass
            #print(self.classerr.value())
        else:
            pass
            #print(None)

        self.reset_meters()

    def save_checkpoint(self, epoch, loss, path):
        """
            Saves a partially/completely trained Torch model

        :param epoch: Epoch number associated with this checkpoint
        :param loss: Current loss associated with this checkpoint
        :return:
        """
        if self.model is not None:
            #print("Saving model checkpoint...")
            if os.path.exists(path):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses,
                    'train_accs': self.train_accs,
                    'test_accs': self.test_accs,
                }, os.path.join(path, str(epoch) + self.checkpoint_ext)) #str(epoch)

    def load_checkpoint(self, path):
        """
            Checks the model path provided initially for a partially/completely trained model
        """
        if os.path.exists(path):
            if len(os.listdir(path)) > 0:

                print("Loading model checkpoint...")
                max         = -1
                checkpoints = glob.glob(os.path.join(path, "*" + self.checkpoint_ext))
                latest      = None

                # Look for latest checkpoint
                for checkpoint in checkpoints:
                    chkpt_filename  = os.path.split(checkpoint)[-1]
                    if not self.checkpoint_ext in chkpt_filename:
                        continue

                    chkpt_filename  = chkpt_filename.replace(self.checkpoint_ext, "")
                    chkpt_num       = int(chkpt_filename)

                    if chkpt_num > max:
                        max     = chkpt_num
                        latest  = checkpoint

                if latest is None:
                    return
                file_name = str(min(int(os.path.splitext(os.path.basename(latest))[0]), self.epochs))
                latest = os.path.join(os.path.dirname(latest), file_name + os.path.splitext(os.path.basename(latest))[1])
                checkpoint = torch.load(latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = int(checkpoint['epoch'])
                self.train_losses = checkpoint['train_losses']
                self.test_losses = checkpoint['test_losses']
                self.train_accs = checkpoint['train_accs']
                self.test_accs = checkpoint['test_accs']
                print("Succesfully loaded checkpoint (epoch {})...".format(self.start_epoch))

    def save_figure(self, path):
        """
            Saves a training\validation classification accuracy curve
                > This function is to be called after performing training

        :param path: Path to save figure
        """
        title = self.__class__.__name__ + " " + self.model_name + " " + self.feat_extractor.dataset_name + " " + self.feat_extractor.__class__.__name__
        if len(self.train_accs) > 0:
            epoch = len(self.train_accs)
            #print("Saving accuracy plot...")
            figure = plt.figure()
            plt.plot(list(range(1, 1 + len(self.train_accs))), self.train_accs)
            plt.xlabel('Epoch')
            plt.ylabel('Average Accuracy [%]')
            legend = ['train']
            if len(self.test_accs) > 0:
                plt.title(title)
                plt.plot(list(range(self.valid_period, 1 + self.valid_period*len(self.test_accs))), self.test_accs)
                legend += ['test']
            plt.legend(legend)

            plt.show()
            figure.savefig(os.path.join(path,str(epoch)+'_accs'))

        if len(self.train_losses) > 0:
            epoch = len(self.train_losses)
            #print("Saving loss plot...")
            figure = plt.figure()
            plt.plot(list(range(1, 1 + len(self.train_losses))), self.train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            legend = ['train']
            if len(self.test_losses) > 0:
                plt.title(title)
                plt.plot(list(range(self.valid_period,
                                    self.valid_period * len(self.test_losses) + 1,
                                    self.valid_period)), self.test_losses)
                legend += ['test']
            plt.legend(legend)

            plt.show()
            figure.savefig(os.path.join(path,str(epoch)+'_losses'))
    def save_log(self, path):
        if len(self.train_accs) > 0 or len(self.test_accs) > 0:
            np.savetxt(os.path.join(path,'accs.csv'), np.array([self.train_accs, self.test_accs]).transpose(), fmt='%.18e', delimiter=',', newline='\n', header='train_accs,test_accs', footer='', comments='', encoding=None)
        if len(self.train_losses) > 0 or len(self.test_losses) > 0:
            np.savetxt(os.path.join(path,'losses.csv'), np.array([self.train_losses, self.test_losses]).transpose(), fmt='%.18e', delimiter=',', newline='\n', header='train_losses,test_losses', footer='', comments='', encoding=None)
        if (len(self.train_losses) > 0 or len(self.test_losses) > 0) and (len(self.train_accs) > 0 or len(self.test_accs) > 0):
            log_file = open(os.path.join(path,'log.txt'), "w")
            par = ''
            for key, val in self.args.items():
                par += str(key) + ': ' + str(val) + '\n'
            log_file.write("best_val_acc: "+str(np.array(self.test_accs).max()) + ' @ '+str(np.argmax(np.array(self.test_accs))+1) +'\n' \
                           + "best_train_acc: "+str(np.array(self.train_accs).max()) + ' @ '+str(np.argmax(np.array(self.train_accs))+1) +'\n'\
                           +"best_val_loss: "+str(np.array(self.test_losses).min()) + ' @ '+str(np.argmin(np.array(self.test_losses))+1) +'\n' \
                           + "best_train_loss: "+str(np.array(self.train_losses).min()) + ' @ '+str(np.argmin(np.array(self.train_losses))+1) +'\n'+
                           par)
            log_file.close()

    @abstractmethod
    def forward_pass(self, sample):
        """
        :param sample: Tuple containing 0: Labels
                                        1: Input features

        :return: A loss object that is used for optimizer updates (e.g. backward pass)
        """
        pass

    @abstractmethod
    def define_model(self, dim_in, train_features=None):
        """
            Defines a network's architecture, used by forward_pass

        :param dim_in: Expected dimension of inputs features

        :return: A Torch "torch.nn.Module" object
        """
        pass


class TorchDatasetIterator(torch.utils.data.Dataset):

    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        size = self.X[0].shape
        data = np.asarray(self.X[i])#.reshape(size[1], size[0])

        if self.transforms:
            data = self.transforms(data).squeeze(0)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data