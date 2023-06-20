from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class ML:

    def __init__(self, dataset, labels, modeltype = 'RF'):

        self.dataset = dataset
        self.labels = labels
        self.modeltype = modeltype

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels)

        if self.modeltype == 'RF':
            self.classifier = RandomForestClassifier()
        if self.modeltype == 'SVM':
            self.classifier = SVC()
        if self.modeltype!= 'RF' and self.modeltype!= 'SVM':
            raise Exception('modeltype must be 'RF' (random forest) or ''SVM'' (support vector machines)')

    def makemodel(self):

        if self.modeltype == 'RF':
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)
        if self.modeltype == 'SVM':
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)

    def accuracy_score(self):
        if self.modeltype == 'RF':
            rf_predictions = self.classifier.predict(self.x_test)
            rf_accuracy = accuracy_score(self.y_test, rf_predictions)
            return rf_accuracy

        if self.modeltype == 'SVM':
            svm_predictions = self.classifier.predict(self.x_test)
            svm_accuracy = accuracy_score(self.y_test, svm_predictions)
            return svm_accuracy

    def predict(self, targets):
        predictions = self.classifier.predict(targets)
        return predictions








