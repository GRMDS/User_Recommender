import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.metrics as metrics
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class evaluation(object):
  def __init__(self, filename):
    final_dict = pd.read_pickle(filename)
    X, y, model = self.cross_validate(final_dict)
    self.y_pred_prob_test, self.y_pred = self.model_evaluate(X,y,model)


  def cross_validate(self,data):
    """Computes cross validation of the results.

    Args: (dictionary) pair_wise dictionary calcualted in user_sim.py

    Returns: (numpy.array) similarity score of each feature for all pairs of users
             (numpy.array) match features for all pairs of users
             Trained Logistic Regression model
    """

    for elem in data:
      score = elem['sim_score']
      key_list = score.keys()
      for key in key_list:
          elem[key] = score[key]
      elem.pop('sim_score')

    pair_df = pd.DataFrame(data)
    col_name = list(pair_df.columns)[3:]

    process_pair_df = pair_df.dropna(thresh = 6)
    process_pair_df = process_pair_df.fillna(0)
    process_pair_df['match'] = pair_df['match'].astype(int)
    X = process_pair_df.iloc[:, 3:].to_numpy()
    y = np.array(process_pair_df['match'])

    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    kfold = model_selection.KFold(n_splits=10)
    cv = model_selection.cross_validate(model, X, y, cv=kfold, scoring='accuracy', return_estimator=True)
    cv_results = cv['test_score']
    best_model = cv['estimator'][np.argmax(cv_results)]
    print('Cross validation results',cv_results.mean())

    return X,y,model

  def model_evaluate(self,X,y,model):
    """Computes evaluation of the results, including calculating confusion metric and ROC plot.

    Args: (numpy.array) similarity score of each feature for all pairs of users
          (numpy.array) match features for all pairs of users
          Trained Logistic Regression model

    Returns: (numpy.array) predicted probability of dependent variable (y) in the test set
             (numpy.array) predicted value of dependent variable (y) in the test set
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train,y_train)
    y_pred_prob_test = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    #Confusion Matrix
    confusion_matrix_census=sklearn.metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)
    tn, fp, fn, tp = confusion_matrix_census.ravel()
    print(confusion_matrix_census)

    #Precision and Recall
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    print('Precision is ',precision,' , Recall is ',recall)

    #f1 score
    f1_score_cen=sklearn.metrics.f1_score(y_test, y_pred,average='binary')
    f1_score_cal=2*precision*recall/(precision+recall)
    print('built-in f1 score is ',f1_score_cen)
    print('calculated f1 score is ',f1_score_cal)



    #Accuracy score and AUC score
    accuracy=accuracy_score(y_test, y_pred)
    print('Accuracy score is ',accuracy)

    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC score is ',roc_auc)


    #ROC
    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return y_pred_prob_test, y_pred


evaluate = evaluation('pairwise_compare_dict.pickle')