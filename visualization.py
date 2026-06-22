import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def show_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.show()
