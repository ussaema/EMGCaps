from . import ClassifierModel
from sklearn.ensemble import RandomForestClassifier
import os

class RandomForest(ClassifierModel):

    def get_class_probabilities(self, test_features):
        return self.classifier.predict_proba(test_features)

    def train(self, args, train_features, train_labels, valid_features = None, valid_labels = None):
        super().train(args, train_features, train_labels, valid_features, valid_labels)
        self.classifier = RandomForestClassifier(n_estimators=self.rf_num_trees)
        self.num_samples = train_features.shape[0]
        if len(train_features.shape) > 2:
            train_features = train_features.reshape(train_features.shape[0], -1)
        self.classifier.fit(train_features, train_labels)
        self.training_acc = self.classifier_accuracy(self.classifier.predict(train_features), train_labels)
        print("Training accuracy:", self.training_acc)
        if valid_features is not None and valid_labels is not None:
            self.test(valid_features, valid_labels)
        self.save_log(self.full_model_path)

    def test(self, test_features, test_labels):
        if len(test_features.shape) > 2:
            test_features = test_features.reshape(test_features.shape[0], -1)
        self.validation_acc = self.classifier_accuracy(self.classifier.predict(test_features), test_labels)
        print("Validation accuracy:", self.validation_acc)
        self.save_log(self.full_model_path)
        return self.validation_acc

    def perform_inference_helper(self, test_features):
        if len(test_features.shape) > 2:
            test_features = test_features.reshape(test_features.shape[0], -1)
        return self.classifier.predict(test_features)

    def save_figure(self, path):
        pass

    def save_log(self, path):
        log_file = open(os.path.join(path, 'log.txt'), "w")
        par = ''
        for key, val in self.args.items():
            par += str(key) + ': ' + str(val) + '\n'
        log_file.write("train_acc:"+ str(self.training_acc) + '\n' \
                       "val_acc:"+ str(self.validation_acc) + '\n' +\
                       par)
        log_file.close()