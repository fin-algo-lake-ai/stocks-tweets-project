# Stocks Tweets Project
The research aimed at finding correlation between tweets and future stock prices.  
Details and results in the [Project Description](2022-11__Stocks_and_Tweets.pdf).  


# File descriptions

## Part 1 (Analysis and Enhancing Results from the Article)

### [Naive_Bayes_from_paper.ipynb](notebooks/Naive_Bayes_from_paper.ipynb)  
Reproduction and improvement results from the related article.


## Part 2 (Introducing New Dataset and Models)

### [nb010_create_labels_for_tweets.ipynb](notebooks/nb010_create_labels_for_tweets.ipynb)  
Prepare new dataset with corrected preprocessing and labeling.

### [nb200_model-ant1-RNN-with-Optuna.ipynb](notebooks/nb200_model-ant1-RNN-with-Optuna.ipynb)  
Draft of train and test RNN (LSTM / GRU) with Optuna tuning. 
Currently not finished yet.


### [nb250_model_catboost_transformer.ipynb](notebooks/nb250_model_catboost_transformer.ipynb)  
Train and test CatBoost and Tranformer (Roberta) models.

### [nb310_model-blend.ipynb](notebooks/nb310_model-blend.ipynb)   
Train and test Naive Bayes model.
Creating ensemble of 3 models (Naive Bayes, CatBoost, Transformer).

