#from bio_embeddings.embed import SeqVecEmbedder
from bio_embeddings.embed import ProtTransBertBFDEmbedder

print("Loading embedder")
embedder = ProtTransBertBFDEmbedder()
#embedder = SeqVecEmbedder()
print("Done")

#

# import torch
# residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
# print(residue_embd)

import random

import pandas as pd

df = pd.read_csv('/home/vprobon/iLIR_EMBED/LIRcentral-OnlyVerified-Canonical.csv', delimiter='\t')
df.rename(columns={
    'UNIPROT ACC':'acc',
    'UNIPROT ID':'id',
    'Motif type':'mtype',
    'Start position':'start',
    'End position': 'end',
    'ExperimentallyVerified(FunctionalYES/NO)':'functional'
    }
    ,inplace=True
)
df['functional']=df['functional'].replace('accessory LIR', 'YES')
# df.head()


df['Up-stream']=df['Up-stream'].fillna("")
df['Down-stream']=df['Down-stream'].fillna("")


def shuffle_string(s, seed):
    random.seed(seed) # For reproducibility
    # Convert the string to a list of characters
    char_list = list(s)
    # Shuffle the list
    random.shuffle(char_list)
    # Convert the list back to a string
    shuffled_string = ''.join(char_list)
    return shuffled_string


def extract_embeddings(df, dims=(10,10),balance=False):
  df_new = df
  emb= [] # List to hold the embeddings
  count=0
  for i in df.iterrows():
    id=i[1]['acc']
    start=i[1]['start']
    end=i[1]['end']
    mtype=i[1]['mtype']
    functional=i[1]['functional']
    up=i[1]["Up-stream"].ljust(10,'X')
    core=i[1]['Motif']
    down=i[1]['Down-stream'].rjust(10,'X')
    up=up[-dims[0]:]
    down=down[0:dims[1]]
    seq=up+core+down
    per_reisdue_embedding = embedder.embed(seq)
    per_protein_embedding = embedder.reduce_per_protein(per_reisdue_embedding)
    emb.append(per_protein_embedding)
    count+=1
  print(f'Recorded {count} embeddings')

  if balance == True:
     # If we balance with oversampling the negative class 
     # we need to also fix the labels
     count2=0
     df_oversample_negative = df[df['functional']=='NO']
     for i in df_oversample_negative.iterrows():
        up=shuffle_string (i[1]["Up-stream"].ljust(10,'X') ,seed = 42)
        # random.shuffle(list(up))
        # up= ''.join(random.shuffle( list(i[1]["Up-stream"].ljust(10,'X') ) ))
        core=i[1]['Motif']
        down = shuffle_string( i[1]['Down-stream'].rjust(10,'X') , seed=42)
        # down = ''.join(random.shuffle(list (i[1]['Down-stream'].rjust(10,'X'))))
        up=up[-dims[0]:]
        down=down[0:dims[1]]
        seq=up+core+down
        per_reisdue_embedding = embedder.embed(seq)
        per_protein_embedding = embedder.reduce_per_protein(per_reisdue_embedding)
        emb.append(per_protein_embedding)  
        count2+=1   
     df_new = pd.concat([df, df_oversample_negative], axis=0)
     print(f'Oversampled {count2} embeddings')
  
  return emb, df_new # Need to return both the embeddings and the dataframe, in case twe oversampled

import numpy as np
def features_from_embeddings(embeddings):
  flattened_features_list = [array.flatten() for array in embeddings]
  flattened_features = np.array(flattened_features_list)
  features_df = pd.DataFrame(flattened_features)
  return features_df


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib # To save classifier models, do parallel computations on multiple CPUs/threads

#from joblib import Parallel, delayed


# Function to train ensemble classifier for a given configuration
def train_ensemble(df,dims, balance):
    print(f'New Ensemble Classifier: flanks {dims}, balance {balance}')
    lir_embeddings, df_new = extract_embeddings(df, dims, balance)
    features_df = features_from_embeddings(lir_embeddings)
    
    if 'functional' in df_new.columns:
        labels = df_new['functional'].values
    
    X = features_df
    y = labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train Base Models
    classifiers = {
        'rf': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}),
        'gb': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}),
        'lr': (LogisticRegression(random_state=42, max_iter=10000), {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}),
        'mlp': (MLPClassifier(random_state=42, max_iter=10000), {'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100), (30,), (10,2)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}),
        'svm': (SVC(probability=True, random_state=42), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']})
    }

    best_estimators = {}
    for name, (clf, params) in classifiers.items():
        grid = GridSearchCV(clf, params, cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_

    ensemble = VotingClassifier(
        estimators=[(name, est) for name, est in best_estimators.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    # Evaluate the Ensemble Model
    y_pred = ensemble.predict(X_test)

    #print(f"Ensemble Model Accuracy (flanks {dims}, balance {balance}):", accuracy_score(y_test, y_pred))
    output = f"Ensemble Model Accuracy (flanks {dims}, balance {balance}):"
    output += str(accuracy_score(y_test, y_pred))
    output += "\n\n"
    #print(classification_report(y_test, y_pred))
    output += str(classification_report(y_test, y_pred))
    #print("Confusion Matrix:")
    output += "Confusion Matrix:\n"
    #print(confusion_matrix(y_test, y_pred))
    output += str(confusion_matrix(y_test, y_pred))
    print(output)

    # Re-train ensemble of best classifiers on full data, 
    # save ensebmle, can later build an ensemble of ensembles!
    ensemble.fit(X, y)
    model_name = f'ensemble_fl_{str(dims[0])}_{str(dims[1])}_bal_{str(balance)}.pkl'
    joblib.dump(ensemble,model_name)
    return ensemble

# List of flanks
#flanks = [(10,10), (5,10), (10,5), (10,7), (7,10), (7,7), (5,5)]
flanks = [(3,10), (10,3), (9,9), (3,3), (10,9), (10,10)]
balance = False

# Number of CPUs to use
#n_jobs = 4

# Train ensemble models in parallel using joblib 
# Some memory leak observed - switch to serial mode ...
# trained_ensembles = Parallel(n_jobs=n_jobs)(delayed(train_ensemble)(df, dims, balance) for dims in flanks)

trained_ensembles=[]
for dims in flanks:
    trained_ensembles.append(train_ensemble(df, dims, balance))

# Now trained_ensembles contains the trained ensemble models for each snapshot
for i, ensemble in enumerate(trained_ensembles):
    print(f"Ensemble {i + 1}:")
    print(ensemble)
    print()




# #### Code below works fine. COde above to use paralellization by joblib

# # flank_max = 10
# # flanks = [(i, j) for i in range(flank_max + 1) for j in range(flank_max + 1)]

# #flanks=[(10,10)]

# flanks = [(10,10), (5,10), (10,5), (10,7), (7,10), (3,10), (10,3), (10,9), (9,10),
#            (5,5), (7,7), (9,9), (3,3)]
# balance=True

# lir_flank_embeddings={}
# for dims in flanks:
#   print(f'New Ensemble Classifier: flanks {flanks}, balance {balance}')  
#   lir_embeddings,  df_new= extract_embeddings(df,dims,balance)
#   features_df = features_from_embeddings(lir_embeddings)
#   if 'functional' in df_new.columns:
#     labels = df_new['functional'].values
#   X = features_df
#   y = labels
#   # Split the data into training and testing sets
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#   label_encoder = LabelEncoder()
#   y_orig = y.copy()
#   y = label_encoder.fit_transform(y)

#   # Split data into training and testing sets
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   ## Train Base Models

#   # Random Forest Classifier
#   rf = RandomForestClassifier(random_state=42)
#   rf_params = {
#       'n_estimators': [100, 200],
#       'max_depth': [None, 10, 20],
#       'min_samples_split': [2, 5]
#   }
#   rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
#   rf_grid.fit(X_train, y_train)


#   # Gradient Boosting Classifier
#   gb = GradientBoostingClassifier(random_state=42)
#   gb_params = {
#       'n_estimators': [100, 200],
#       'learning_rate': [0.01, 0.1],
#       'max_depth': [3, 5, 7]
#   }
#   gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc')
#   gb_grid.fit(X_train, y_train)


#   # Logistic Regression
#   lr = LogisticRegression(random_state=42, max_iter=10000)
#   lr_params = {
#       'C': [0.01, 0.1, 1, 10],
#       'penalty': ['l2'],
#       'solver': ['lbfgs']
#   }
#   lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc')
#   lr_grid.fit(X_train, y_train)

#   # Multilayer Perceptron Classifier
#   mlp = MLPClassifier(random_state=42, max_iter=10000)
#   mlp_params = {
#       'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100), (30,), (10,2)], 
#       'activation': ['relu', 'tanh'],
#       'alpha': [0.0001, 0.001, 0.01]
#   }
#   mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='roc_auc')
#   mlp_grid.fit(X_train, y_train)

#   # Support Vector Machine Classifier
#   svm = SVC(probability=True, random_state=42)
#   svm_params = {
#       'C': [0.1, 1, 10],
#       'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
#       'gamma': ['scale', 'auto']
#   }
#   svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc')
#   svm_grid.fit(X_train, y_train)

#   # Combine Models using Voting Classifier
#   ensemble = VotingClassifier(
#       estimators=[
#           ('rf', rf_grid.best_estimator_),
#           ('gb', gb_grid.best_estimator_),
#           ('lr', lr_grid.best_estimator_),
#           ('mlp', mlp_grid.best_estimator_),
#           ('svm', svm_grid.best_estimator_)
#       ],
#       voting='soft'
#   )
#   ensemble.fit(X_train, y_train)
#   # Evaluate the Ensemble Model
#   y_pred = ensemble.predict(X_test)

#   print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
#   print(classification_report(y_test, y_pred))
#   print("Confusion Matrix:")
#   print(confusion_matrix(y_test, y_pred))

#   # Re-rrain ensemble of best classifiers on full data, 
#   # save ensebmle, can later build an ensemble of ensembles!
#   ensemble.fit(X, y)
#   model_name = f'ensemble_fl_{str(dims[0])}_{str(dims[1])}_bal_{str(balance)}.pkl'
#   joblib.dump(ensemble,model_name)




# # count = 0
# # lir_embeddings=[] # per protein (segment) embeddings
# # for i in df.iterrows():
# #   id=i[1]['acc']
# #   start=i[1]['start']
# #   end=i[1]['end']
# #   mtype=i[1]['mtype']
# #   functional=i[1]['functional']
# #   up=i[1]["Up-stream"].ljust(10,'X')
# #   core=i[1]['Motif']
# #   down=i[1]['Down-stream'].rjust(10,'X')
  
# #   seq=up+core+down
# #   per_reisdue_embedding = embedder.embed(seq)
# #   per_protein_embedding = embedder.reduce_per_protein(per_reisdue_embedding)
# #   lir_embeddings.append(per_protein_embedding)
# #   count+=1
# # print(f'Recorded {count} embeddings')

# # import numpy as np
# # flattened_features_list = [array.flatten() for array in lir_embeddings]
# # flattened_features = np.array(flattened_features_list)
# # features_df = pd.DataFrame(flattened_features)


# #### Now build classifiers
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# X = features_df
# y = labels

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the classifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# # Make predictions and evaluate
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f'Accuracy: {accuracy:.4f}')

# k_folds = KFold(n_splits = 10, shuffle=True, random_state=42)

# scores = cross_val_score(clf, X, y, cv = k_folds)
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score

# multilayerperceptron = MLPClassifier(solver='lbfgs', random_state=10, max_iter=1000)

# parameters = {
#     'hidden_layer_sizes': [(30,), (10,2)]
# }

# classifiers = GridSearchCV(multilayerperceptron, parameters, cv=3, scoring="roc_auc")
# classifiers.fit(X_train, y_train)
# classifier = classifiers.best_estimator_

# predicted_testing_labels = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, predicted_testing_labels)

# print(f"Our model has an accuracy of {accuracy:.2}")
# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test,predicted_testing_labels)


# import matplotlib.pyplot as plt


# from sklearn import svm
# from sklearn.metrics import RocCurveDisplay, auc, PrecisionRecallDisplay, DetCurveDisplay
# from sklearn.model_selection import StratifiedKFold



# def validate_clf(classifier,cv, name, params, inputsize):
#   tprs = []
#   aucs = []
#   mean_fpr = np.linspace(0, 1, 100)

#   fig, ax = plt.subplots(figsize=(6, 6))
#   for fold, (train, test) in enumerate(cv.split(X, y)):
#       X_train = X[train]
#       y_train = y[train]
#       classifier.fit(X[train], y[train])
#       viz = RocCurveDisplay.from_estimator(
#           classifier,
#           X[test],
#           y[test],
#           name=f"ROC fold {fold}",
#           alpha=0.3,
#           lw=1,
#           ax=ax,
#           # the following does not work, need to check module versions
#           #plot_chance_level=(fold == n_splits - 1),
#       )
#       interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#       interp_tpr[0] = 0.0
#       tprs.append(interp_tpr)
#       aucs.append(viz.roc_auc)

#   mean_tpr = np.mean(tprs, axis=0)
#   mean_tpr[-1] = 1.0
#   mean_auc = auc(mean_fpr, mean_tpr)
#   std_auc = np.std(aucs)
#   ax.plot(
#       mean_fpr,
#       mean_tpr,
#       color="b",
#       label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#       lw=2,
#       alpha=0.8,
#   )

#   std_tpr = np.std(tprs, axis=0)
#   tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#   tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#   ax.fill_between(
#       mean_fpr,
#       tprs_lower,
#       tprs_upper,
#       color="grey",
#       alpha=0.2,
#       label=r"$\pm$ 1 std. dev.",
#   )

#   ax.set(
#       xlabel="False Positive Rate",
#       ylabel="True Positive Rate",
#       title=f"{name} ('{params}') Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
#   )
#   ax.legend(loc="lower right")
#   plt.show()

# target_names=['Non-functional LIR', "Functional LIR"]
# X = np.array(features_df)
# y = labels


# # from sklearn.preprocessing import StandardScaler
# # scaler = StandardScaler()
# # X=scaler.fit_transform(X)


# n_splits = 3
# cv = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=42)



# kernels=["linear","poly","rbf","sigmoid"]
# for kernel in kernels:
#   classifier = svm.SVC(kernel=kernel, probability=True, random_state=42)
#   validate_clf(classifier,cv, 'svm',str(kernel),24)

# from sklearn.neighbors import KNeighborsClassifier
# for nns in [3,4,7,9,11,13,15,21,31,61]:
#   classifier = KNeighborsClassifier(n_neighbors=nns)
#   classifier.fit(X, y)
#   validate_clf(classifier,cv,'kNN',nns,24)


# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0).fit(X, y)
# validate_clf(classifier,cv,'Logistic regression','(defaults)',24)


# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score


# label_encoder = LabelEncoder()
# y_orig = y.copy()
# y = label_encoder.fit_transform(y)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # Train Base Models

# # Random Forest Classifier
# rf = RandomForestClassifier(random_state=42)
# rf_params = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5]
# }
# rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
# rf_grid.fit(X_train, y_train)


# # Gradient Boosting Classifier
# gb = GradientBoostingClassifier(random_state=42)
# gb_params = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5, 7]
# }
# gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc')
# gb_grid.fit(X_train, y_train)


# # Logistic Regression
# lr = LogisticRegression(random_state=42, max_iter=10000)
# lr_params = {
#     'C': [0.01, 0.1, 1, 10],
#     'penalty': ['l2'],
#     'solver': ['lbfgs']
# }
# lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc')
# lr_grid.fit(X_train, y_train)

# # Multilayer Perceptron Classifier
# mlp = MLPClassifier(random_state=42, max_iter=10000)
# mlp_params = {
#     'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
#     'activation': ['relu', 'tanh'],
#     'alpha': [0.0001, 0.001, 0.01]
# }
# mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='roc_auc')
# mlp_grid.fit(X_train, y_train)

# # Support Vector Machine Classifier
# svm = SVC(probability=True, random_state=42)
# svm_params = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# }
# svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc')
# svm_grid.fit(X_train, y_train)

# # Combine Models using Voting Classifier
# ensemble = VotingClassifier(
#     estimators=[
#         ('rf', rf_grid.best_estimator_),
#         ('gb', gb_grid.best_estimator_),
#         ('lr', lr_grid.best_estimator_),
#         ('mlp', mlp_grid.best_estimator_),
#         ('svm', svm_grid.best_estimator_)
#     ],
#     voting='soft'
# )
# ensemble.fit(X_train, y_train)


# # Evaluate the Ensemble Model
# y_pred = ensemble.predict(X_test)

# print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
