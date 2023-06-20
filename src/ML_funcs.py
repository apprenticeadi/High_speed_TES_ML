from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

class ML:

    def __init__(self, dataset, labels, modeltype = 'RF'):

        self.dataset = dataset
        self.labels = labels
        self.modeltype = modeltype

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels)

        if self.modeltype == 'RF':
            self.classifier = RandomForestClassifier()

        elif self.modeltype == 'SVM':
            self.classifier = SVC()

        elif self.modeltype == 'BDT':
            self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
            self.dtest = xgb.DMatrix(self.x_test, label=self.y_test)
        else:
            raise Exception('modeltype must be "RF", "SVM" or  "BDT" (Random forest, support vector machines or boosted decision tree)')

    def makemodel(self, num_rounds = 10):

        if self.modeltype == 'RF':
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)
        if self.modeltype == 'SVM':
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)
        if self.modeltype == 'BDT':
            params = {
                'max_depth': 3,
                'eta': 0.1,
                'num_class': 10,
                'objective': 'multi:softmax'
            }
            self.classifier = xgb.train(params, self.dtrain, num_rounds)


    def accuracy_score(self):

            predictions = self.classifier.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            return accuracy


    def predict(self, targets):
        if self.modeltype == 'BDT':
            d_matrix = xgb.DMatrix(targets)
            predictions = self.classifier.predict(d_matrix)
            return predictions.astype(int)
        else:
            predictions = self.classifier.predict(targets)
            return predictions








