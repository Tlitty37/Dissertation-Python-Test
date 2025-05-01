import pandas as pd #this allows me to read the excel file I am importing
import numpy as np ## graph creation and simplicity working with arrays
import matplotlib.pyplot as plt #plotting the data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, precision_recall_curve, 
                            ConfusionMatrixDisplay, roc_auc_score)

from sklearn.pipeline import make_pipeline

#other imports were chosen while I was writinig the script, they are all necessary for the program to run
def datapreparation(xlsx_file):
    df=pd.read_excel(xlsx_file) #opens excel file
    ncolumns = ['Polarity', 'Subjectivity', 'Mode', 'Set_Type'] #these are the columns necessary for the SVM to run, ni this step I am ensuring they are here
    columncheck=[col for col in ncolumns if col not in df.columns]
    if columncheck:
        print('Not all columns are present. You are missing : {columncheck}')
        raise ValueError
    df = df.dropna(subset=ncolumns) #getting rid of empty rows
    trainingset=df[df['Set_Type']=='Training'] #assigns all 'Training' to the training set
    testset=df[df['Set_Type']=='Test']

    if len(trainingset)<10 or len(testset)<10:
        print('Not many samples, ths may effect data accuracy. Mention this in report.')
    return trainingset, testset

def dothekNN(trainingsentiment, labeltraining):
    pipeline=make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'))   ###change k here
    pipeline.fit(trainingsentiment, labeltraining)
    return pipeline

def analyse(model, testsentiment, labeltest, le):
    labelprediction=model.predict(testsentiment)
    labelprobability=model.predict_proba(testsentiment)[:,1] if hasattr(model, 'predict_proba') else None
    ltdecoded=le.inverse_transform(labeltest)
    lpdecoded=le.inverse_transform(labelprediction)
    print('kNN Accuracy: {accuracy_score(labeltest, labelprediction):.2f}')
    print('Classification Report: Genres Reggae')
    print(classification_report(ltdecoded, lpdecoded, zero_division=0))
    print('Confusion Matrix:')
    confmatrix = confusion_matrix(ltdecoded, lpdecoded, labels=le.classes_)
    disp=ConfusionMatrixDisplay(confusion_matrix=confmatrix, display_labels=le.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix:')
    plt.show()
    if len(le.classes_)==2:
        print(f'\nROC AUC Score: {roc_auc_score(labeltest, labelprobability):.3f}')
        precision, recall, _ =precision_recall_curve(labeltest, labelprobability)
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recaall')
        plt.ylabel('Precisiion')
        plt.title('Precision-Recall Curve:')
        plt.show()
    
def visualise(model, trainingsentiment, labeltraining, testsentiment, labeltest, le):
    plt.figure(figsize=(16, 9), dpi=100)
    plt.style.use('seaborn-v0_8')
    class_colors = {'Major': '#FF6B6B', 'Minor': '#4ECDC4'}

    x_min, x_max = trainingsentiment[:, 0].min() - 0.1, trainingsentiment[:, 0].max() + 0.1
    y_min, y_max = trainingsentiment[:, 1].min() - 0.1, trainingsentiment[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Get predictions as class indices
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create a colormap from the unique classes
    unique_classes = le.classes_
    colors = ['#FF6B6B', '#4ECDC4'][:len(unique_classes)]
    
    plt.contourf(xx, yy, Z, 
                 colors=colors, 
                 alpha=0.15, 
                 levels=len(unique_classes)-1)
    plt.contour(xx, yy, Z, 
                colors='black', 
                linewidths=1, 
                linestyles='--', 
                levels=[0.5])
    
    # Plot training points
    for i, cls in enumerate(unique_classes):
        mask_train = (le.inverse_transform(labeltraining) == cls)
        plt.scatter(trainingsentiment[mask_train, 0], trainingsentiment[mask_train, 1],
                   c=colors[i],
                   label=f'Training {cls}',
                   marker='o',
                   edgecolor='k',
                   s=100,
                   alpha=0.9)
    
    # Plot test points
    labelprediction = model.predict(testsentiment)
    for i, cls in enumerate(unique_classes):
        mask_test = (le.inverse_transform(labelprediction) == cls)
        plt.scatter(testsentiment[mask_test, 0], testsentiment[mask_test, 1],
                   c=colors[i],
                   label=f'Test Predicted {cls}',
                   marker='s',
                   edgecolor='k',
                   s=120,
                   alpha=0.7)
    
    plt.xlabel('Polarity (Standardized)', 
              fontsize=14, 
              fontweight='bold', 
              labelpad=10)
    plt.ylabel('Subjectivity (Standardized)',
               fontsize=14,
               fontweight='bold',
               labelpad=10)
    plt.title('kNN Decision Boundary (k=5)\nPolarity vs. Subjectivity World',
              fontsize=16,
              fontweight='bold',
              pad=20)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left',
                        borderaxespad=0.,
                        fontsize=12,
                        title_fontsize='13',
                        title='Legend',
                        frameon=True)
    legend.get_frame().set_facecolor('#F5F5F5')
    legend.get_frame().set_edgecolor('black')
    plt.grid(True, 
             linestyle='--', 
             alpha=0.5)
    
    plt.tight_layout()
    plt.show()




xlsx_file="/Users/tobylodge/Downloads/Training and Test Data Dissertation/genre_splits/testtrain_Reggae.xlsx"

trainingset, testset = datapreparation(xlsx_file)

trainingsentiment=trainingset[['Polarity', 'Subjectivity']].values
testsentiment=testset[['Polarity', 'Subjectivity']].values
le=LabelEncoder()
labeltraining=le.fit_transform(trainingset['Mode']) #this one uses fit_transform as it is the training set
labeltest = le.transform(testset['Mode']) #if ths used fit_transform it would just create a new line, obviously don't want this

model=dothekNN(trainingsentiment, labeltraining)
analyse(model, testsentiment, labeltest, le)
visualise(model, trainingsentiment, labeltraining, testsentiment, labeltest, le)
