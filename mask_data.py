import pandas as pd
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
from faker import Faker
from random import shuffle, sample, randint, random
import ast
import re


class artificial_data(object):

    def __init__(self):
        # Train a data2text model to generate artificial description of project on the website
        file_name = 'dataset_des.txt'
        self.ai_text_generator(file_name)
        # Load the trained model
        self.ai_text = aitextgen(model_folder="trained_model", tokenizer_file="aitextgen.tokenizer.json")

        # Use Faker package to generate fake users' information
        self.fake = Faker()

        # Mask users' features
        all_user_features = pd.read_csv('user_feature.csv',index_col=0)
        all_user_features = self.user_feature_mask(all_user_features)
        all_user_features = self.clean_all_user_features(all_user_features)
        all_user_features.to_csv('fake_all_user_features.csv')

        # Mask users' activities features
        # Join the users' activities datasets into one dataframe as match_df
        datasets_list = ['top_click_all.csv','visits_30_user.csv','course_enrolments.csv','order_info.csv']
        match_df = self.concat_match_df(datasets_list,all_user_features)
        match_df = self.match_df_mask(match_df)
        match_df.to_csv('match_df.csv')



    def ai_text_generator(self,file_name):
        """Train a data to text model to generate artificial text for project description, and save the model
        as file.

        Args: file name of input text dataset

        Returns: None
        """

        # Train a custom BPE Tokenizer on the downloaded text
        # This will save one file: `aitextgen.tokenizer.json`, which contains the
        # information needed to rebuild the tokenizer.
        train_tokenizer(file_name)
        tokenizer_file = "aitextgen.tokenizer.json"

        # GPT2ConfigCPU is a mini variant of GPT-2 optimized for CPU-training
        config = GPT2ConfigCPU()

        # Instantiate aitextgen using the created tokenizer and config
        ai = aitextgen(tokenizer_file=tokenizer_file, config=config)

        # Build datasets for training by creating TokenDatasets,
        # which automatically processes the dataset with the appropriate size.
        data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)

        # Train the model. It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
        # On a 2019 Macbook Pro, this took ~50 minutes to run.
        ai.train(data, batch_size=8, num_steps=50000, generate_every=5000, save_every=5000)
        return None



    def user_feature_mask(self,all_user_features):
        """Masks the user's information with fake data. Meanwhile, computes resampling of data
        because of the sparsity of the dataset.

        Args: (pandas.dataframe) users features data

        Returns: (pandas.dataframe) masked users features data
        """

        # fake username of the email address
        all_user_features['email'] = [self.fake.user_name()+ '@' + all_user_features['email'].values[i].split('@')[1]
                                    for i in range(0,all_user_features.shape[0])]

        # fake username, first_name, last_name
        all_user_features['username'] = [self.fake.user_name() for _ in range(0,all_user_features.shape[0])]
        all_user_features['first_name'] = [self.fake.first_name() for _ in range(0,all_user_features.shape[0])]
        all_user_features['last_name'] = [self.fake.last_name() for _ in range(0,all_user_features.shape[0])]
        
        # shuffle and resample company, job_title, education, university, major
        all_user_features['company'] = self.shuffle_list(list(all_user_features['company']))
        all_user_features['job_title'] = sample(list(all_user_features['job_title'])+
                                        list(set(all_user_features['job_title']))+
                                        [self.fake.job() for _ in range(40)],k=all_user_features.shape[0])
        all_user_features['education'] = sample(list(all_user_features['education'])+
                                        list(set(all_user_features['education']))*20,
                                        k=all_user_features.shape[0])
        all_user_features['university'] = sample(list(all_user_features['university'])+
                                        list(set(all_user_features['university']))*5,
                                        k=all_user_features.shape[0])
        all_user_features['major'] = sample(list(all_user_features['major'])+
                                        list(set(all_user_features['major']))*5,
                                        k=all_user_features.shape[0])
        
        # shuffle cols 11 - 40, 42 - 47
        for i in list(range(11,41))+list(range(42,48)):
            all_user_features.iloc[:,i] = self.shuffle_list(list(all_user_features.iloc[:,i]))

        # project_list, randomly append 200 project information in the null rows
        all_user_features[['project_list']] = all_user_features[['project_list']].applymap(self.literal_return)
        notnull_row_origin = list(all_user_features.loc[all_user_features['project_list'].notnull()].index)
        self.random_project_list(all_user_features,200)
        
        # num_follower, append random num to those users who is appended random project_list on previous step
        notnull_row = list(all_user_features.loc[all_user_features['project_list'].notnull()].index)
        append_row = list(set(notnull_row)-set(notnull_row_origin))
        all_user_features.loc[append_row,'num_follower'] = [randint(1,50) for _ in range(len(append_row))]

        # impact score, append impact score to those users who is appended random project_list on previous step
        # impact score range: 0 - 10
        all_user_features.loc[append_row,'impact_score'] = [random()*10 for _ in range(len(append_row))]
        
        return all_user_features


    def shuffle_list(self,df_col):
        shuffle(df_col)
        return df_col

    def random_project_list(self,all_user_features,num):
        """ Creats augmented project list for null value in features of project_list.

        Args: (pandas.dataframe) users features data
              (int) number of rows that need to append project list

        Returns: None
        """

        # Random num of project in each list (range: 1-10)
        # project id is random number from 0 to 2000
        # project title and description generated by ai text generater, title with max_length 15
        # description with max_length 150
        # programing language is randomly chosen from a list of programing software
        programing = ['Python','R','C++','C','Other','Java',None]
        null_row = list(all_user_features.loc[all_user_features['project_list'].isnull()].index)
        random_append_index = sample(null_row,k = num)
        for i in random_append_index:
            project_num = randint(1,10)
            project_list = []
            for _ in range(project_num):
                info = (randint(0,2000),self.ai_text.generate(1,max_length = 15,return_as_list=True)[0],
                        self.ai_text.generate(1,max_length = 150,return_as_list=True)[0],
                        sample(programing,k=1)[0])
                project_list.append(info)
            all_user_features.loc[i,'project_list'] = project_list
        return None

    def clean_all_user_features(self,all_user_features):
        # Clean the text data and turn them into lower case
        clean_list = ['company','job_title','education','major','university']
        for l in clean_list:
            all_user_features[l] = all_user_features[l].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
        return all_user_features

    def literal_return(val):
        # Turn the string into list syntax
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            return None



    def concat_match_df(self,datasets_list,all_user_features):
        """Combine several dataframes of user behavior into one, and clean the dataset.

        Args: (list) list of user behavior datasets
              (pandas.dataframe) users features data

        Return: (pandas.dataframe) dataframe with user behavior data
        """

        match_df = pd.DataFrame()
        for file in datasets_list:
            df = pd.read_csv(file, index_col=0)
            match_df = pd.concat([match_df,df],axis=1)

        # Select uid corresponding with all_user_features
        match_df = match_df.loc[list(set(all_user_features.index)&set(match_df.index)),:]

        # Clean the text data
        clean_list = ['all_enrol','all_visits','all_orders']
        for l in clean_list:
            match_df[[l]] = match_df[[l]].applymap(self.str_to_list)
        return match_df

    def str_to_list(string):
        # Turn the string into list syntax
        if str(string) not in ['NaN','nan','None']:
            return string.strip('][').split(', ')
        else:
            return None

    def match_df_mask(self,match_df):
        """Computes resampling of the match data.

        Args: (pandas.dataframe) dataframe with user behavior data

        Returns: (pandas.dataframe) dataframe with augmented user behavior data
        """
        
        # expand purchase list
        order_num = [randint(200,300) for _ in range(50)]
        shuffle_order = list(set(sample(list(match_df[match_df.all_orders.isnull()].index),k=170)))
        match_df.loc[shuffle_order,'all_orders'] = sample(order_num*5,k=len(shuffle_order))
        match_df.loc[shuffle_order,'purchased_entity'] = match_df.loc[shuffle_order,'all_orders']

        # expand visits list
        shuffle_visit = list(set(sample(list(match_df[match_df.all_visits.isnull()].index),k=280)))
        visit_info = sample(list(match_df[match_df.all_visits.notnull()].index)*7,k=len(shuffle_visit))
        for col in ['url','time','time_period','all_visits']:
            match_df.loc[shuffle_visit,col] = list(match_df.loc[visit_info,col])

        # expand enrollment
        shuffle_enrol = list(set(sample(list(match_df[match_df.all_enrol.isnull()].index),k=160)))
        enrol_info = list(set(match_df[match_df.all_enrol.notnull()]['enrolment']))
        match_df.loc[shuffle_enrol,'enrolment'] = sample(enrol_info*7,k=len(shuffle_enrol))
        match_df.loc[shuffle_enrol,'all_enrol'] = match_df.loc[shuffle_enrol,'enrolment']
        return match_df


fake_data = artificial_data()
