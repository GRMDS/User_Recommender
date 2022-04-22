from pyjarowinkler import distance   
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import pandas as pd
import numpy as np
import ast
import re
from ast import literal_eval
from itertools import combinations
import pickle
import warnings
warnings.filterwarnings('ignore')

# Methods for generate similarities
class recom_user(object):
    def __init__(self):
        self.interest_list = ['Fraud_Detection', 'Risk_Scoring', 'Healthcare', 'Internet_Search', 'Marketing_Effectiveness',
                              'Website_Recommendations', 'Image_Recognition', 'Speech_Recognition',
                              'Airline_Route_Planning', 'Price_Analytics', 'Supply_Chain_Optimization',
                              'Talent_Acquisition_Analytics', 'Environment_Analytics', 'Epidemiology',
                              'Social_Policy', 'Evaluation_and_Assessment']
        self.vectorizer = TfidfVectorizer()
        
    def compare(self, info1, info2):
        """Computes similarity comparison of each features.

        Args: (pandas.dataframe) row in all_user_features dataframe

        Returns: (dictionary) results of similarity score of each features
        """

        results = {}
        
        #email extension sim
        results['email_sim'] = self.email_extension(info1['email'], info2['email'])
        
        #University company job_title education major sim
        user_info_list = ['university','company','job_title','education','major']
        for l in user_info_list:
            results[l] = self.simple_text(info1[l], info2[l])
        
        # interest sim
        results['interest_sim'] = self.cos_sim([info1[x] if info1[x] != None else 0 for x in self.interest_list], 
                                               [info2[x] if info2[x] != None else 0 for x in self.interest_list])
        # followers
        results['follower_sim'] = info2['num_follower']

        # impact score
        results['impact_score_sim'] = info2['impact_score']        

        # recency
        results['recency_sim'] = info2['recency']

        # news competition
        competition_list = ['covid_competition','news_competition','property_competition','restaurant_competition',
                            'investment_competition']
        for l in competition_list:
            results[l] = info1[l] & info2[l]

        # project part
        if (info1['project_list']==None or info2['project_list']==None or 
            len(info1['project_list']) ==0 or len(info2['project_list'])==0):
            results['proj_title_sim'],results['proj_des_sim'],results['proj_language_sim'] = None, None, None
            return results

        
        # project title sim
        title_list1 = [x[1] for x in info1['project_list']]
        title_list2 = [x[1] for x in info2['project_list']]
        pairs = list(itertools.product(title_list1, title_list2))
        results['proj_title_sim'] = sum(map(lambda x : self.simple_text(x[0],x[1]), pairs)) / len(pairs)
        
        # project language sim
        lan_list1 = [x[3] for x in info1['project_list'] if x[3] != None]
        lan_list2 = [x[3] for x in info2['project_list'] if x[3] != None]
        if len(lan_list1) == 0 or len(lan_list2) == 0:
            results['proj_language_sim'] = None
        else:
            pairs = list(itertools.product(lan_list1, lan_list2))
            results['proj_language_sim'] = sum(map(lambda x : x[0].lower()==x[1].lower(), pairs)) / len(pairs) * 100
            
        # project description sim
        des_list1 = [x[2] for x in info1['project_list'] if x[2] != None]
        des_list2 = [x[2] for x in info2['project_list'] if x[2] != None]
        if len(des_list1) == 0 or len(des_list2) == 0:
            results['proj_des_sim'] = None
        else:
            results['proj_des_sim'] = self.long_text(des_list1, des_list2)
        
        return results
        
    def email_extension(self,u1,u2):
        # Judge if these two email address have the same suffix
        if u1 in [None,''] or u2 in [None,'']:
            return None
        try:
            return (u1.split('@')[1].split('.')[1] == u2.split('@')[1].split('.')[1]) * 100
        except:
            return None
        
    def simple_text(self,u1,u2):
        if u1 in [None,''] or u2 in [None,'']:
            return None
        #u1 = re.sub(r'[^\w]', '', u1).lower()
        #u2 = re.sub(r'[^\w]', '', u2).lower()
        return distance.get_jaro_distance(u1, u2) * 100
    
    def cos_sim(self,u1,u2):
    #   return cosine_similarity([u1], [u2])[0][0] * 100
        u1 = np.array(u1)
        u2 = np.array(u2)
        # numpy's cosine similarity
        cosine_sim = np.dot(u1,u2) / (np.sqrt(np.dot(u1,u1)) * np.sqrt(np.dot(u2,u2)))
        if np.isnan(cosine_sim):
          return None
        else:
          return cosine_sim * 100
    
    def long_text(self,doc1,doc2):
        # Vectorize long text information and calculate their cosine similarity score
        #vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(doc1+doc2)
        dense = vectors.todense()
        dense1 = dense[:len(doc1)]
        dense2 = dense[len(doc1):]
        pairs = list(itertools.product(dense1, dense2))
        return sum(map(lambda x : cosine_similarity(x[0], x[1])[0][0] , pairs)) / len(pairs) * 100
    


# Generate dependent varibles, as matched/unmatched
# Matching or not is decided by wheather or not the users have some common activities:
# Visiting a same project/webinar/course/other pages, Taking a same order...

class user_sim_evaluator(object):
    def __init__(self, n_node, n_webinar,match_file,feature_file):
        match_df = pd.read_csv(match_file,index_col=0)
        all_user_features = pd.read_csv(feature_file,index_col=0)
        all_user_features = self.clean_features(all_user_features)
        match_df = self.clean_match(match_df)
        self.recom = recom_user()
        self.pair_wise, self.common_ids = self.evaluate_all(n_node, n_webinar,all_user_features,match_df)
        
    def evaluate_all(self, n_node, n_webinar, all_user_features,match_df):
        """Calculates all pairs of user similarity score and the matching between users in pairs.

        Args: n_node (int), n_webinar (int) minimum number of project and webinar that user commonly visit would 
              be viewed as matched
              all_user_features (pandas.dataframe) users features dataset
              match_df (pandas.dataframe) users activities dataset

        Returns: pair_wise (dictionary) similarity score of each feature for all pairs of user
                 common_id (list) valid user id intersect with user id in users features dataset
        """
        
        # Filter out users with at least one visit in project/webinar
        valid_list = match_df[(match_df.node.str.find('/node/')>-1) | (match_df.webinar.str.find('/webinar/')>-1)]
        # Turn the string of node and webinar into list syntax
        valid_list[['node', 'webinar']] = valid_list[['node', 'webinar']].applymap(literal_eval)
        valid_ids = valid_list.index.to_list()
        common_ids = list(set(valid_ids).intersection(all_user_features.index))
        # In case ids are mixed

        # formatting features for evaluation 
        filter = all_user_features.index.isin(common_ids)
        all_features = all_user_features[filter]
        all_features = all_features.where(pd.notnull(all_features), None)
        valid_list = valid_list.where(pd.notnull(valid_list), None)
              

        pair_wise = []
        count = 0
        for tuples in combinations(common_ids, 2):
            count += 1
            r1 = valid_list.loc[tuples[0],:]
            r2 = valid_list.loc[tuples[1],:]
            match = []
            match.append(not set(r1['node'][:n_node]+r1['webinar'][:n_webinar]).isdisjoint(r2['node'][:n_node]+r2['webinar'][:n_webinar]))

            # Checking common in courses, orders, visits
            activities_list = ['all_enrol','all_orders','all_visits']
            for l in activities_list:
              activity_1 = valid_list.loc[tuples[0],l]
              activity_2 = valid_list.loc[tuples[1],l]
              if activity_1 == None or activity_2 ==None:
                match.append(False)
              elif len(set(activity_1)&set(activity_2))>0:
                match.append(True)
              else:
                match.append(False)

            
            # If there is one match, users have common activity
            match = sum(match)
            if match >1: 
                match = 1 
            uinfo_1 = all_features.loc[tuples[0],:]
            uinfo_2 = all_features.loc[tuples[1],:]
            sim_score = self.recom.compare(uinfo_1,uinfo_2)                                                                                            
            pair_wise.append({'user1': tuples[0], 'user2': tuples[1], 'match': match, 'sim_score': sim_score})

        # Create a pickle file to save the data
        with open("pairwise_compare_dict.pickle", "wb",) as myFile:
            pickle.dump(pair_wise, myFile)
    
        return pair_wise, common_ids

    def literal_return(self,val):
        # Turn the string into list syntax
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            return None

    def str_to_list(self,string):
        # Turn the string into list syntax
        if str(string) not in ['NaN','nan','None']:
            return string.strip('][').split(', ')
        else:
            return None

    def clean_features(self,all_user_features):
        # Clean the text data
        all_user_features[['project_list']] = all_user_features[['project_list']].applymap(self.literal_return)
        clean_list = ['company','job_title','education','major','university']
        for l in clean_list:
            all_user_features[l] = all_user_features[l].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
        return all_user_features

    def clean_match(self,match_df):
        clean_list = ['all_enrol','all_visits','all_orders']
        for l in clean_list:
            match_df[[l]] = match_df[[l]].applymap(self.str_to_list)
        return match_df


user_sim = user_sim_evaluator(5,3,'match_df.csv','fake_all_user_features.csv') # Numbers up to change
pair_wise = user_sim.pair_wise
print(len(pair_wise))