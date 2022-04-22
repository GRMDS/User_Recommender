import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
import user_sim
import ast
import re

class model_retrain(object):
    def __init__(self,final_dict,all_user_features):
        model, res = self.model_retrain_test(final_dict)
        model.to_csv('model_weights.csv')
        res.to_csv('model_accuracy.csv')

        self.recom = user_sim.recom_user()
        self.user_sim_table = self.update_user_recom_table(all_user_features, model)
        self.user_sim_table.to_csv('user_sim_table.csv')

    def model_retrain_test(self,data):
        """Computes Logistic Regression to initially calculate the total similarity of each pair of users.
           Independent variables of LR model are similarity scores of each feature, dependent variable is
           match feature in dictionary calculated in user_sim.py.
           
        Args: (dictionary) pair_wise dictionary calcualted in user_sim.py

        Returns: (pandas.dataframe) model parameters, weight of each feature to calculate the total similarity 
                 score in next step
                 (pandas.dataframe) model results
        """

        for elem in data:
            score = elem['sim_score']
            key_list = score.keys()
            for key in key_list:
                elem[key] = score[key]
            elem.pop('sim_score')
        
        pair_df = pd.DataFrame(data)
        col_name = list(pair_df.columns)[3:]
        
        # Clean the pairwise data
        process_pair_df = pair_df.dropna(thresh = 6)
        process_pair_df = process_pair_df.fillna(0)
        process_pair_df['match'] = pair_df['match'].astype(int)
        x = process_pair_df.iloc[:, 3:].to_numpy()
        y = np.array(process_pair_df['match'])
        
        # Utilize LR model to get the weight of each similarity features
        # Dependent variables are users' match
        # Independent variables are users' similarity scores
        model = LogisticRegression(solver='liblinear', multi_class='ovr')
        kfold = model_selection.KFold(n_splits=10)
        cv = model_selection.cross_validate(model, x, y, cv=kfold, scoring='accuracy', return_estimator=True)
        cv_results = cv['test_score']
        best_model = cv['estimator'][np.argmax(cv_results)]
        
        # Generate results df
        results_table = pd.DataFrame()
        results_table['accuracy'] = cv_results
        results_table['timestamp'] = pd.Timestamp.today()
        best_model.fit(x,y)
        model_para = np.absolute(best_model.coef_[0])
        model_para = model_para / model_para.sum()
        model_para = pd.DataFrame(zip(col_name, model_para), columns = ['features','weights'])
        model_para['timestamp'] = pd.Timestamp.today()
            
        return model_para, results_table

        # Periodically update the recommendation score table
        # Calculate total similarity score of each user by using the weights in LR model
    def update_user_recom_table(self,data, model):
        """Computes algorithm to update the recommendation table. 

        Args: (pandas.dataframe) users features dataset
              (pandas.dataframe) weights of each feature to calculate the total score

        Returns: (pandas.dataframe) the recommendation table for all users
        """

        results = []
        count = 0
        user_list = data.index.to_list()
        weight = np.array(model['weights'])
        for user1 in user_list:
            sim_user = []
            for user2 in user_list:
                if user1 != user2:
                    count += 1
                    if count%100000 == 0:
                        print(count)
                    compare = self.recom.compare(data.loc[user1,:], data.loc[user2,:])
                    compare = {key:0 if str(value) in ['nan','None','NaN'] else value for key,value in compare.items()}
                    # Calculate total similarity score of each user by using the weights in LR model
                    score = np.dot(np.array(list(compare.values())),weight)
                    sim_user.append({'user': user1, 'sim_user': user2, 'sim_score':score})
            # Users with top 10 total similarity score will be recommended
            sim_user = sorted(sim_user, key = lambda x: x['sim_score'], reverse = True)[:10] # Could be 50
            results.extend(sim_user)
        results = pd.DataFrame(results)
        return results

    def clean_features(self,all_user_features):
        # Clean the text data
        all_user_features[['project_list']] = all_user_features[['project_list']].applymap(self.literal_return)
        clean_list = ['company','job_title','education','major','university']
        for l in clean_list:
            all_user_features[l] = all_user_features[l].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
        return all_user_features
    
    def literal_return(self,val):
        # Turn the string into list syntax
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            return None

final_dict = pd.read_pickle('pairwise_compare_dict.pickle')
all_user_features = pd.read_csv('fake_all_user_features.csv',index_col=0)
retrain = model_retrain(final_dict, all_user_features)