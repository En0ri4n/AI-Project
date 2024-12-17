from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score


class StatisticsHelper:
    def __init__(self, X, y, model, y_test, _pred):
        self.X = X
        self.y = y
        self.model = model
        self.y_test = y_test
        self.y_pred = _pred

    def show_accuracy(self):
        print("---- Accuracy ----")
        print(f'Accuracy: {accuracy_score(self.y_test, self.y_pred)}')
        print("------------------")

    def show_roc_auc_score(self):
        print("---- ROC AUC Score ----")
        print(f'ROC AUC Score: {roc_auc_score(self.y_test, self.y_pred)}')
        print("------------------------")

    def show_classification_report(self):
        print("---- Classification Report ----")
        print(f'{classification_report(self.y_test, self.y_pred)}')
        print("-------------------------------")

    def show_confusion_matrix(self):
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
        scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy')
        print("--- Cross Validation Scores ---")
        print(f'Cross-validated scores: {scores}')
        print(f'Mean accuracy: {scores.mean()}')
        print(f'Standard deviation: {scores.std()}')
        print("-------------------------------")
