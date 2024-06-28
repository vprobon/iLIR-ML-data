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
df.head()


df['Up-stream']=df['Up-stream'].fillna("")
df['Down-stream']=df['Down-stream'].fillna("")

count = 0
lir_embeddings=[] # per protein (segment) embeddings
for i in df.iterrows():
  id=i[1]['acc']
  start=i[1]['start']
  end=i[1]['end']
  mtype=i[1]['mtype']
  functional=i[1]['functional']
  up=i[1]["Up-stream"].ljust(10,'X')
  core=i[1]['Motif']
  down=i[1]['Down-stream'].rjust(10,'X')
  
  seq=up+core+down
  per_reisdue_embedding = embedder.embed(seq)
  per_protein_embedding = embedder.reduce_per_protein(per_reisdue_embedding)
  lir_embeddings.append(per_protein_embedding)
  count+=1
print(f'Recorded {count} embeddings')

import numpy as np
flattened_features_list = [array.flatten() for array in lir_embeddings]
flattened_features = np.array(flattened_features_list)
features_df = pd.DataFrame(flattened_features)

df['functional']=df['functional'].replace('accessory LIR', 'YES')
if 'functional' in df.columns:
    labels = df['functional'].values

#### Now build classifiers
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
X = features_df
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')

k_folds = KFold(n_splits = 10, shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

multilayerperceptron = MLPClassifier(solver='lbfgs', random_state=10, max_iter=1000)

parameters = {
    'hidden_layer_sizes': [(30,), (10,2)]
}

classifiers = GridSearchCV(multilayerperceptron, parameters, cv=3, scoring="roc_auc")
classifiers.fit(X_train, y_train)
classifier = classifiers.best_estimator_

predicted_testing_labels = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predicted_testing_labels)

print(f"Our model has an accuracy of {accuracy:.2}")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predicted_testing_labels)


import matplotlib.pyplot as plt


from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc, PrecisionRecallDisplay, DetCurveDisplay
from sklearn.model_selection import StratifiedKFold



def validate_clf(classifier,cv, name, params, inputsize):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

  fig, ax = plt.subplots(figsize=(6, 6))
  for fold, (train, test) in enumerate(cv.split(X, y)):
      X_train = X[train]
      y_train = y[train]
      classifier.fit(X[train], y[train])
      viz = RocCurveDisplay.from_estimator(
          classifier,
          X[test],
          y[test],
          name=f"ROC fold {fold}",
          alpha=0.3,
          lw=1,
          ax=ax,
          # the following does not work, need to check module versions
          #plot_chance_level=(fold == n_splits - 1),
      )
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
      lw=2,
      alpha=0.8,
  )

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
  )

  ax.set(
      xlabel="False Positive Rate",
      ylabel="True Positive Rate",
      title=f"{name} ('{params}') Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
  )
  ax.legend(loc="lower right")
  plt.show()

target_names=['Non-functional LIR', "Functional LIR"]
X = np.array(features_df)
y = labels


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X=scaler.fit_transform(X)


n_splits = 5
cv = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=42)



kernels=["linear","poly","rbf","sigmoid"]
for kernel in kernels:
  classifier = svm.SVC(kernel=kernel, probability=True, random_state=42)
  validate_clf(classifier,cv, 'svm',str(kernel),24)
