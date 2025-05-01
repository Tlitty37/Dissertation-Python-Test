import pandas as pd #this allows me to read the excel file I am importing
import numpy as np ## graph creation and simplicity working with arrays
import matplotlib.pyplot as plt #plotting the data
from sklearn.svm import SVC #scikit has a selection of librariers for SVC, I chose this one as the data is not linearly seperable and this should not be too coomputationally taxing on my computer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, precision_recall_curve, 
                            average_precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
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

def necadj(n_samples, smallcratio):
 
    if n_samples <= 10:  # Only for Classical/World genres (4 samples)
        C = 10.0  # Reduced from 100 to prevent over-regularization
        gamma = 'scale'  # Default behavior (was 0.5)
        class_weight = 'balanced'  # Use sklearn's balanced weighting instead of 1.0/minority_ratio
    else:
        # Original settings that gave you 62% accuracy
        C = 1.0
        gamma = 'scale'
        class_weight = 'balanced'
    
    return {'C': C, 'gamma': gamma, 'class_weight': class_weight}

def dotheSVM(trainingsentiment, labeltraining, smallcratio):
    n_samples = len(labeltraining)
    params = necadj(n_samples, smallcratio)
    
    # Keep your original pipeline structure
    pipeline = make_pipeline(
        StandardScaler(), 
        SVC(
            kernel='rbf',
            C=params['C'],
            gamma=params['gamma'],
            class_weight=params['class_weight'],
            probability=True,
            random_state=42
        )
    )
    
    # Your original grid search parameters
    param_grid = {
        'svc__C': [0.1, 1, 10],  # Original range
        'svc__gamma': ['scale', 'auto']  # Original options
    }
    
    # Use standard 5-fold CV (no min(n_samples) to match original behavior
    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # Original setting
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    model.fit(trainingsentiment, labeltraining)
    print(f"Selected parameters - C: {model.best_params_['svc__C']}, gamma: {model.best_params_['svc__gamma']}")
    return model.best_estimator_

def analyse(model, testsentiment, labeltest, le):
    labelprediction=model.predict(testsentiment)
    labelprobability=model.predict_proba(testsentiment)[:,1]
    ltdecoded=le.inverse_transform(labeltest)
    lpdecoded=le.inverse_transform(labelprediction)
    print("Best MODEL: {model.get_params()['svc']}")
    print('Classfication report: Genres-World:')
    print(classification_report(ltdecoded, lpdecoded, zero_division=0))
    print('Confusion Matrix:')
    print(confusion_matrix(ltdecoded, lpdecoded))
    if len(le.classes_)==2:
        print(f'\nROC AUC Score: {roc_auc_score(labeltest, labelprobability):.3f}')
        print(f'Average Precision Score: {average_precision_score(labeltest, labelprobability):.3f}')
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
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),np.linspace(y_min, y_max, 500))
 
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    plt.contourf(xx, yy, Z, colors=['#FF6B6B', '#4ECDC4'], alpha=0.15, levels=1)
    
    plt.contour(xx, yy, Z, 
               colors='black', 
               linewidths=1, 
               linestyles='--', 
               levels=[0.5])

    for cls in le.classes_:
        mask_train = (le.inverse_transform(labeltraining) == cls)
        plt.scatter(trainingsentiment[mask_train, 0], trainingsentiment[mask_train, 1],
                   c=class_colors[cls], 
                   label=f'Training {cls}',
                   marker='o', 
                   edgecolor='k',
                   s=100,
                   alpha=0.9)
        

        mask_test = (le.inverse_transform(labeltest) == cls)
        plt.scatter(testsentiment[mask_test, 0], testsentiment[mask_test, 1],
                   c=class_colors[cls],
                   label=f'Test {cls} (Predicted)',
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
    plt.title('Decision Boundary: Polarity vs. Subjectivity\nModel Performance by Genre, World',
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
    
xlsx_file="/Users/tobylodge/Downloads/Training and Test Data Dissertation/genre_splits/testtrain_World.xlsx"

trainingset, testset = datapreparation(xlsx_file)

trainingsentiment=trainingset[['Polarity', 'Subjectivity']].values
testsentiment=testset[['Polarity', 'Subjectivity']].values
le=LabelEncoder()
labeltraining=le.fit_transform(trainingset['Mode']) #this one uses fit_transform as it is the training set
labeltest = le.transform(testset['Mode']) #if ths used fit_transform it would just create a new line, obviously don't want this
smallcratio = trainingset['Mode'].value_counts(normalize=True).min()
model=dotheSVM(trainingsentiment, labeltraining, smallcratio)
analyse(model, testsentiment, labeltest, le)
visualise(model, trainingsentiment, labeltraining, testsentiment, labeltest, le)
