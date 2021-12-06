#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


POSTGRES = {

'user' : 'postgres',
'pw' : 'admin',
'host' : 'localhost',
'port' : '5432',
'db' : 'soccerdata'
}


# In[3]:


class Config(object):
    # ...
    SQLALCHEMY_DATABASE_URI =  os.environ.get('DATABASE_URL') or 'postgresql://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
    SQLALCHEMY_TRACK_MODIFICATIONS = False

config = Config()

cnx = create_engine(config.SQLALCHEMY_DATABASE_URI)


# In[4]:


df = pd.read_sql_query('''
SELECT f.*, l.event_date, l.home_goals, l.away_goals, l.home_team, l.home_team_id, l.away_team, l.away_team_id 
FROM fixture f left join league l on l.fixture_id = f.fixture_id;''', cnx)


# In[5]:


df


# In[6]:


df.rename(columns={'home_goals': 'goals_home', 'away_goals': 'goals_away'}, inplace=True)

conditions = [(df['goals_away'] < df['goals_home']), (df['goals_away'] == df['goals_home']), (df['goals_away'] > df['goals_home'])]
values_home = ['W', 'D', 'L']
values_away = ['L', 'D', 'W']

df['home_wld'] = np.select(conditions, values_home)
df['away_wld'] = np.select(conditions, values_away)

df = df.sort_values('event_date')
df.reset_index(inplace=True, drop=True)


# In[7]:


def get_goals_scored(df, team):
    mask_home = np.where(df['home_team'] == team)
    mask_away = np.where(df['away_team'] == team)
    
    goals_scored_home = df.loc[mask_home[0], ['event_date','goals_home']].rename(columns={'goals_home': 'goals_scored'})
    goals_scored_away = df.loc[mask_away[0], ['event_date','goals_away']].rename(columns={'goals_away': 'goals_scored'})

    goals_scored = pd.concat([goals_scored_home, goals_scored_away]).sort_values(by='event_date')

    return goals_scored
    
def get_goals_taken(df, team):
    
    mask_home = np.where(df['home_team'] == team)
    mask_away = np.where(df['away_team'] == team)
    
    goals_taken_away = df.loc[mask_away[0],  ['event_date','goals_home']].rename(columns={'goals_home': 'goals_taken'})
    goals_taken_home = df.loc[mask_home[0],  ['event_date','goals_away']].rename(columns={'goals_away': 'goals_taken'})

    goals_taken = pd.concat([goals_taken_home, goals_taken_away]).sort_values(by='event_date')
    
    return goals_taken


# In[8]:


def expanding_average_goals(df):
    
    df.loc[:, ['average_goals_scored_home', 'average_goals_taken_home', 'average_goals_scored_away', 'average_goals_taken_away']] = 0
    
    for team in np.unique(list(df['home_team'].values) + list(df['away_team'].values)):
        
        team_mask = np.where((df['home_team'] == team) | (df['away_team'] == team))
        df_team = df.loc[team_mask[0], :]
        df_team.reset_index(inplace=True, drop=False)

        res_taken = get_goals_taken(df_team, team)
        res_taken = res_taken['goals_taken'].expanding(1).mean().shift()

        res_scored = get_goals_scored(df_team, team)
        res_scored = res_scored['goals_scored'].expanding(1).mean().shift()

        df_team.loc[np.where(df_team['home_team'] == team)[0], 'is_home'] = True
        df_team.loc[np.where(df_team['home_team'] != team)[0], 'is_home'] = False
        df_team.loc[np.where(df_team['away_team'] == team)[0], 'is_away'] = True
        df_team.loc[np.where(df_team['away_team'] != team)[0], 'is_away'] = False

        df_team.loc[df_team['is_home'].values,'average_goals_scored_home'] = res_scored[df_team['is_home'].values].values
        df_team.loc[df_team['is_home'].values,'average_goals_taken_home'] = res_taken[df_team['is_home'].values].values

        df_team.loc[df_team['is_away'].values,'average_goals_scored_away'] = res_scored[df_team['is_away'].values].values
        df_team.loc[df_team['is_away'].values,'average_goals_taken_away'] = res_taken[df_team['is_away'].values].values

        df.loc[df_team['index'], 'average_goals_scored_home'] = df_team.loc[:, 'average_goals_scored_home'].values
        df.loc[df_team['index'], 'average_goals_taken_home'] = df_team.loc[:, 'average_goals_taken_home'].values
        df.loc[df_team['index'], 'average_goals_scored_away'] = df_team.loc[:, 'average_goals_scored_away'].values
        df.loc[df_team['index'], 'average_goals_taken_away'] = df_team.loc[:, 'average_goals_taken_away'].values
        
    return df


# In[9]:


def get_stat(df, team, stat):
    mask_home = np.where(df['home_team'] == team)
    mask_away = np.where(df['away_team'] == team)
    
    if ('_home' in stat) or ('_away' in stat):
        stat_home = df.loc[df.index.intersection(mask_home[0]), stat]
        stat_away = df.loc[df.index.intersection(mask_away[0]), stat]
    else:
        stat_home = df.loc[df.index.intersection(mask_home[0]), stat + '_home']
        stat_away = df.loc[df.index.intersection(mask_away[0]), stat + '_away']

    stat_team = pd.concat([stat_home, stat_away])
    
    return stat_team


# In[10]:


def expanding_average_stat(df, stat):
    
    for team in np.unique(list(df['home_team'].values) + list(df['away_team'].values)):
        team_mask = np.where((df['home_team'] == team) | (df['away_team'] == team))
        df_team = df.loc[team_mask[0], :]
        df_team.reset_index(inplace=True, drop=False)

        res, res_home, res_away = pd.Series(get_stat(df_team, team, stat))
        res = res.expanding(1).mean().shift()
        res_home = res.expanding(1).mean().shift()
        res_away = res_away.expanding(1).mean().shift()

        df_team.loc[np.where(df_team['home_team'] == team)[0], 'is_home'] = True
        df_team.loc[np.where(df_team['home_team'] != team)[0], 'is_home'] = False
        df_team.loc[np.where(df_team['away_team'] == team)[0], 'is_away'] = True
        df_team.loc[np.where(df_team['away_team'] != team)[0], 'is_away'] = False

        if ('_home' not in stat) and ('_away' not in stat):
            home_stat_label = stat + '_home'
            away_stat_label = stat + '_away'
        else:
            home_stat_label = stat
            away_stat_label = stat
        
        df_team.loc[df_team['is_home'].values, 'overall_average_' + home_stat_label] = res[df_team['is_home'].values].values
        df_team.loc[df_team['is_away'].values, 'overall_average_' + away_stat_label] = res[df_team['is_away'].values].values

        df_team.loc[df_team['is_home'].values, 'home_average_' + home_stat_label] = res[df_team['is_home'].values].values
        df_team.loc[df_team['is_away'].values, 'away_average_' + away_stat_label] = res[df_team['is_away'].values].values
        
        df_team.loc[df_team['is_home'].values, 'overall_average_' + home_stat_label] = res[df_team['is_home'].values].values
        df_team.loc[df_team['is_away'].values, 'overall_average_' + away_stat_label] = res[df_team['is_away'].values].values

        df.loc[df_team['index'], 'overall_average_' + home_stat_label] = df_team.loc[:, 'overall_average_' + home_stat_label].values
        df.loc[df_team['index'], 'overall_average_' + away_stat_label] = df_team.loc[:, 'overall_average_' + away_stat_label].values
        
    return df


# In[11]:


def get_stat(df, team, stat):
    mask_home = np.where(df['home_team'] == team)
    mask_away = np.where(df['away_team'] == team)
    
#     if ('_home' in stat) or ('_away' in stat):
    stat_home = df[stat].reindex(index=mask_home[0])
    stat_away = df[stat].reindex(index=mask_away[0])
#     else:
#         stat_home = df.loc[mask_home[0], stat + '_home']
#         stat_away = df.loc[mask_away[0], stat + '_away']

    stat_team = pd.concat([stat_home, stat_away])
    
    return stat_team, stat_home, stat_away


# In[23]:


team = 'Manchester United'
stat = 'shots_on_goal_home'
team_mask = np.where((df['home_team'] == team) | (df['away_team'] == team))
df_team = df.loc[team_mask[0], :]
df_team.reset_index(inplace=True, drop=False)

res, res_home, res_away = pd.Series(get_stat(df_team, team, stat))
res = res.expanding(1).mean().shift()


# In[25]:


res_home


# In[24]:


cols = ['shots_on_goal_home', 'shots_on_goal_away','shots_off_goal_home', 'shots_off_goal_away', 'total_shots_home',
        'total_shots_away', 'blocked_shots_home', 'blocked_shots_away','shots_insidebox_home', 'shots_insidebox_away', 
        'shots_outsidebox_home', 'shots_outsidebox_away', 'fouls_home', 'fouls_away', 'corner_kicks_home', 'corner_kicks_away', 
        'offsides_home', 'offsides_away', 'ball_possession_home', 'ball_possession_away', 'yellow_cards_home', 
        'yellow_cards_away', 'red_cards_home', 'red_cards_away', 'goalkeeper_saves_home', 'goalkeeper_saves_away', 
        'total_passes_home', 'total_passes_away', 'passes_accurate_home', 'passes_accurate_away', 'passes_perc_home', 
        'passes_perc_away']

df = expanding_average_goals(df)

for col in cols:
    df = expanding_average_stat(df, col)
    
df.dropna(axis=0, how='any', inplace=True)


# ## Classification

# In[20]:


features = ['overall_average_goals_scored_home', 'overall_average_goals_taken_home', 'overall_average_goals_scored_away', 
            'overall_average_goals_taken_away', 'overall_average_shots_on_goal_home', 'overall_average_shots_on_goal_away',
            'overall_average_shots_off_goal_home', 'overall_average_shots_off_goal_away', 'overall_average_total_shots_home', 
            'overall_average_total_shots_away', 'overall_average_blocked_shots_home', 
        'average_blocked_shots_away', 'average_shots_insidebox_home', 'average_shots_insidebox_away', 
        'average_shots_outsidebox_home', 'average_shots_outsidebox_away', 'average_fouls_home', 'average_fouls_away', 
        'average_corner_kicks_home', 'average_corner_kicks_away', 'average_offsides_home', 'average_offsides_away', 
        'average_ball_possession_home', 'average_ball_possession_away', 'average_yellow_cards_home', 
        'average_yellow_cards_away', 'average_red_cards_home', 'average_red_cards_away', 'average_goalkeeper_saves_home', 
        'average_goalkeeper_saves_away', 'average_total_passes_home', 'average_total_passes_away', 
        'average_passes_accurate_home', 'average_passes_accurate_away', 'average_passes_perc_home', 'average_passes_perc_away']

df['home_wld'] = LabelEncoder().fit_transform(df['home_wld'])
X = df[features + ['home_wld']]
y = X['home_wld']
X = df[features]

n = X.shape[0]
idx_train = int(np.round(2*n/3,0))
y_train = y[:idx_train]
y_test = y[idx_train:]

X_train = X.iloc[:idx_train,:]
X_test = X.iloc[idx_train:,:]


# In[27]:


### Unbalanced dataset

print('% of home wins: ', len(np.where(y_train == 0)[0]) / len(y_train))
print('% of draws: ', len(np.where(y_train == 1)[0]) / len(y_train))
print('% of home loss: ', len(np.where(y_train == 2)[0]) / len(y_train))


# In[295]:


features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = GaussianNB()

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


# In[296]:


accuracy_score(y_pred, y_test)


# In[297]:


# target_pipeline = Pipeline(steps=[('categorical', LabelEncoder())])
# targetCol = 'home_wld'
features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = LinearSVC(max_iter=10000,C=0.01)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[298]:


accuracy_score(y_pred, y_test)


# In[299]:


# target_pipeline = Pipeline(steps=[('categorical', LabelEncoder())])
# targetCol = 'home_wld'
features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = RandomForestClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[300]:


accuracy_score(y_pred, y_test)


# In[308]:


param_search = { 
    'model__n_estimators': [200, 500],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth' : [4,5,6,7,8],
    'model__criterion' :['gini', 'entropy']
}

features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = RandomForestClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])

tscv = TimeSeriesSplit(n_splits=4)
gsearch = GridSearchCV(estimator=pipeline, cv=tscv,
                        param_grid=param_search)
gsearch.fit(X_train, y_train)


# In[309]:


print("Best parameter (CV score=%0.3f):" % gsearch.best_score_)
print(gsearch.best_params_)


# In[310]:


features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = RandomForestClassifier(max_depth=gsearch.best_params_['model__max_depth'], 
                             max_features=gsearch.best_params_['model__max_features'], 
                             n_estimators=gsearch.best_params_['model__n_estimators'])
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
pipeline.fit(X_train, y_train)


# In[311]:


y_pred = pipeline.predict(X_test)
print(accuracy_score(y_pred, y_test))


# In[316]:


param_search = { 
    'model__C': np.arange(10**-3, 10**1, 10**-1),
    'model__penalty': ['l1', 'l2'],
}

features_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('features', features_pipeline, features)])

clf = LinearSVC(dual=False)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])

tscv = TimeSeriesSplit(n_splits=4)
gsearch = GridSearchCV(estimator=pipeline, cv=tscv,
                        param_grid=param_search)
gsearch.fit(X, y)
y_pred = pipeline.predict(X_test)


# In[317]:


print("Best parameter (CV score=%0.3f):" % gsearch.best_score_)
print(gsearch.best_params_)

