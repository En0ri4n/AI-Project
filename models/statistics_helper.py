from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


class StatisticsHelper:
    """
    A class to help with displaying statistics for classification and regression models.
    """
    def __init__(self, X, y, model, y_test, _pred):
        self.X = X
        self.y = y
        self.model = model
        self.y_test = y_test
        self.y_pred = _pred

    def show_accuracy(self):
        """
        Display the accuracy of the model.
        """
        print("---- Accuracy ----")
        print(f'Accuracy: {accuracy_score(self.y_test, self.y_pred)}')
        print("------------------")

    def show_roc_auc_score(self):
        """
        Display the ROC AUC score of the model.
        """
        print("---- ROC AUC Score ----")
        print(f'ROC AUC Score: {roc_auc_score(self.y_test, self.y_pred)}')
        print("------------------------")

    def show_classification_report(self):
        """
        Display the classification report of the model.
        """
        print("---- Classification Report ----")
        print(f'{classification_report(self.y_test, self.y_pred)}')
        print("-------------------------------")

    def show_confusion_matrix(self):
        """
        Display the confusion matrix of the model.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.yticks([0, 1], ['No', 'Yes'])
        plt.title('Confusion Matrix')
        # Annotate the confusion matrix with the counts
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')

        plt.show()

    def show_cross_val_score(self):
        """
        Display the cross-validated scores of the model.
        """
        scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy')
        print("--- Cross Validation Scores ---")
        print(f'Cross-validated scores: {scores}')
        print(f'Mean accuracy: {scores.mean()}')
        print(f'Standard deviation: {scores.std()}')
        print("-------------------------------")

    def show_regression_statistics(self):
        """
        Display the regression statistics of the model.
        """
        print("---- Regression Statistics ----")
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        print("-----------------------------")

    def show_all(self):
        """
        Display all the statistics.
        """
        self.show_accuracy()
        self.show_roc_auc_score()
        self.show_classification_report()
        self.show_confusion_matrix()
        self.show_cross_val_score()
        self.show_regression_statistics()
