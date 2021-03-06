# User Recommender of RMDS LAB
RMDS Lab is aiming to make its work more public to let data scientists enjoy the charm in algorithms. This is one of its open-source projects, user recommender, which
is deployed on the [RMDS LAB](https://grmds.org).

## Introduction
RMDS wants to recommend RMDS user to other RMDS users based on user similarity. RMDS would collect user data from the database and calculate out similarity scores for each pair of users, and recommend users to users who have high similarity scores with them.

The User Recommender consists of four components: data processing, model development, model update, and model evaluation.
- [mask_data.py](https://github.com/GRMDS/User_Recommender/blob/main/mask_data.py): Preparing data for recommender algorithm. In this section, we have three main steps. First, to protect user privacy, we mask users' information with fake data. then, because of the sparsity of the data, we fill the null data with multiple methods. finally, we integrate and clean the data.
- [user_sim.py](https://github.com/GRMDS/User_Recommender/blob/main/user_sim.py): Computing algorithm to Calculate user similarity score of each features for all users using various methods.
- [model_retrain.py](https://github.com/GRMDS/User_Recommender/blob/main/model_retrain.py): Utilize Logistic Regression model to initially calculate the total similarity score for all pairs of users. Then, the parameters of the model are used as weights, which are used to calculate the total similarity score each time the recommendation table is updated.
- [evaluation.py](https://github.com/GRMDS/User_Recommender/blob/main/evaluation.py): Computing cross validation of the results and evaluating the results by confusion metric and ROC plot.
- [quick_start.ipynb](https://github.com/GRMDS/User_Recommender/blob/main/quick_start.ipynb): This is a Jupyter notebook of the complete process of the recommendation system. It can provide an overview of the system and easy to implement each step.


## Requirements of development environment
- pyjarowinkler 1.8
- scikit-learn 0.24.0

## License
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
