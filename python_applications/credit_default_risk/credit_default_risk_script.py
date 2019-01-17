import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import gc
import time


# Useful function for reducing memory usage of a dataframe (found it on the interwebs!)
def reduce_memory(df):
    """Reduce memory usage of a dataframe by setting data types. """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df


params = {
    'boosting_type': 'goss',
    'n_estimators': 10000,
    'learning_rate': 0.005,
    'num_leaves': 52,
    'max_depth': 12,
    'subsample_for_bin': 240000,
    'reg_alpha': 0.45,
    'reg_lambda': 0.48,
    'colsample_bytree': 0.5,
    'min_split_gain': 0.025,
    'subsample': 1
}


def lightgbm_predict(df_all, categorical_feat, rounds=150, n_folds=10):
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=666)

    df_train = df_all[df_all['TARGET'].notnull()].drop(columns=['SK_ID_CURR'])
    df_test = df_all[df_all['TARGET'].isnull()].drop(columns=['TARGET'])

    # Spliting training data into features/labels
    app_train_proc_feat, app_train_proc_lbl = df_train.loc[:, df_train.columns != 'TARGET'], df_train.loc[:, 'TARGET']

    avg_score = 0

    temp_results = np.zeros(df_test.shape[0])

    for n_fold, (train_idx, cv_idx) in enumerate(folds.split(app_train_proc_feat, app_train_proc_lbl)):
        # Train data
        train_feats = app_train_proc_feat.iloc[train_idx]
        train_lbls = app_train_proc_lbl.iloc[train_idx]

        # Cross-validation data
        cv_feats = app_train_proc_feat.iloc[cv_idx]
        cv_lbls = app_train_proc_lbl.iloc[cv_idx]

        # Features
        feats = app_train_proc_feat.columns

        # LightGBM classifier
        lgbm = LGBMClassifier(**params)

        start = time.time()
        # Fitting the model
        lgbm.fit(train_feats, train_lbls, eval_set=[(train_feats, train_lbls), (cv_feats, cv_lbls)],
                 eval_metric='auc', verbose=400, early_stopping_rounds=rounds,
                 feature_name=list(feats), categorical_feature=categorical_feat)
        end = time.time()

        # Testing the model on the CV data
        preds = lgbm.predict_proba(cv_feats, num_iteration=lgbm.best_iteration_)[:, 1]

        temp_results += lgbm.predict_proba(df_test.drop(columns=['SK_ID_CURR']), num_iteration=lgbm.best_iteration_)[:, 1] / n_folds

        # Scores
        roc_auc = roc_auc_score(cv_lbls, preds)
        avg_score += (roc_auc / n_folds)

        print('Training time: %.2f seconds' % (end - start))
        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc))

        del lgbm, train_feats, train_lbls, cv_feats, cv_lbls
        gc.collect()

    df_test['TARGET'] = temp_results

    print('Average score on folds: %.6f' % avg_score)

    return avg_score, df_test[['SK_ID_CURR', 'TARGET']]


# Application

print('Application')

application_train = reduce_memory(pd.read_csv('./datasets/application_train.csv'))
application_test = reduce_memory(pd.read_csv('./datasets/application_test.csv'))
application_all = pd.concat([application_train,application_test], sort=True)

del application_train, application_test
gc.collect()


application_all = application_all[application_all['CODE_GENDER'] != 'XNA']
application_all = application_all[application_all['NAME_FAMILY_STATUS'] != 'Unknown']
application_all['DAYS_EMPLOYED'] = application_all['DAYS_EMPLOYED'].replace(365243, np.NaN)
application_all = application_all[(application_all.AMT_INCOME_TOTAL < 10000000)]

categorical_features=[
    'FLAG_OWN_CAR',
    'CODE_GENDER',
    'FLAG_OWN_REALTY',
    'NAME_CONTRACT_TYPE',
    'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY',
    'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY'
]

application_all_proc = application_all[['SK_ID_CURR','TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE','OWN_CAR_AGE'] + categorical_features]

for col in categorical_features:
    application_all_proc.loc[:,col], uniques = pd.factorize(application_all_proc[col])



def apply_agg(df_to_agg, cols_agg_funcs, group_by_col, prefix=''):
    new_feats = []
    for col, funcs in cols_agg_funcs.items():
        for f in funcs:
            if prefix:
                col_name = '{}_{}_{}'.format(prefix,col,f.upper())
            else:
                col_name = '{}_{}'.format(col,f.upper())
            new_feats.append(col_name)
            _df = df_to_agg.groupby([group_by_col])[col].agg(f).reset_index(name=col_name)
            df_to_agg = pd.merge(df_to_agg, _df, on=group_by_col, how='left')
            del _df
            gc.collect()
    return df_to_agg, new_feats


docs = [cl for cl in application_all.columns if 'FLAG_DOCUMENT_' in cl]
application_all_proc['DOC_COUNT'] = application_all[docs].replace(np.nan,0).sum(axis=1)
application_all_proc['EXT_SOURCES_PROD_CBRT'] = (application_all_proc['EXT_SOURCE_1']*application_all_proc['EXT_SOURCE_2']*application_all_proc['EXT_SOURCE_3'])**(1/3)
for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
    feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
    application_all_proc[feature_name] = eval('np.{}'.format(function_name))(
        application_all_proc[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)


application_all_proc['YEARS_TO_PAY'] = application_all_proc['AMT_CREDIT'] / application_all_proc['AMT_ANNUITY']
application_all_proc['DOWN_PAYMENT'] = application_all['AMT_GOODS_PRICE']-application_all_proc['AMT_CREDIT']
application_all_proc['CREDIT_TO_GOODS_RATIO'] = application_all_proc['AMT_CREDIT']/application_all['AMT_GOODS_PRICE']
application_all_proc['ANNUITY_TO_INCOME_RATIO'] = application_all_proc['AMT_ANNUITY'] / application_all_proc['AMT_INCOME_TOTAL']
application_all_proc['EMPLOYED_TO_BIRTH_RATIO'] = application_all_proc['DAYS_EMPLOYED'] / application_all_proc['DAYS_BIRTH']
application_all_proc['CAR_TO_BIRTH_RATIO'] = application_all['OWN_CAR_AGE'] / application_all_proc['DAYS_BIRTH']
application_all_proc['CAR_TO_EMPLOYED_RATIO'] = application_all_proc['OWN_CAR_AGE'] / application_all_proc['DAYS_EMPLOYED']


del application_all
gc.collect()


# Bureau
print('Bureau')

bureau = reduce_memory(pd.read_csv('./datasets/bureau.csv').rename(index=str, columns={'AMT_ANNUITY': 'AMT_ANNUITY_BUREAU'}))

proc_df = bureau.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].agg('count').reset_index(name='BURO_APP_CNT')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = bureau[bureau.CREDIT_ACTIVE=='Closed'].groupby(['SK_ID_CURR']).size().reset_index(name='CLOSED_CREDIT_CNT')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = bureau[bureau.CREDIT_ACTIVE=='Active'].groupby(['SK_ID_CURR']).size().reset_index(name='ACTIVE_CREDIT_CNT')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = bureau[bureau.CREDIT_ACTIVE=='Sold'].groupby(['SK_ID_CURR']).size().reset_index(name='SOLD_CREDIT_CNT')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = bureau[bureau.CREDIT_ACTIVE=='Active'].groupby(['SK_ID_CURR']).agg({'AMT_ANNUITY_BUREAU': np.sum}).reset_index().rename(columns={'AMT_ANNUITY_BUREAU': 'AMT_ANNUITY_BUREAU_ACTIVE'}, inplace=False)
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = bureau.groupby(['SK_ID_CURR']).agg({'DAYS_CREDIT':np.max}).reset_index().rename(columns={'DAYS_CREDIT': 'DAYS_MOST_RECENT_CREDIT'}, inplace=False)
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

bureau['CREDIT_DURATION'] = bureau['DAYS_CREDIT_ENDDATE']-bureau['DAYS_CREDIT']
bureau['DEBT_CREDIT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY_BUREAU']

cols_agg = {
    'DAYS_CREDIT': ['min', 'mean', 'std'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean', 'std'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean', 'sum'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'AMT_ANNUITY_BUREAU': ['max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['mean'],
    'CREDIT_DURATION': ['min', 'max', 'mean', 'std','sum'],
    'DEBT_CREDIT_PERCENTAGE': ['min', 'max', 'mean', 'std'],
    'DEBT_CREDIT_DIFF': ['min', 'max', 'mean', 'std','sum'],
    'CREDIT_TO_ANNUITY_RATIO': ['min', 'max', 'mean', 'std','sum']
}


bureau, bureau_stat_agg_features = apply_agg(bureau, cols_agg, 'SK_ID_CURR')
application_all_proc = reduce_memory(pd.merge(application_all_proc, bureau[['SK_ID_CURR']+bureau_stat_agg_features], on='SK_ID_CURR', how='left'))

del bureau, proc_df
gc.collect()


application_all_proc.fillna({'CLOSED_CREDIT_CNT': 0, 'BAD_DEBT_CNT':0, 'ACTIVE_CREDIT_CNT':0, 'SOLD_CREDIT_CNT':0, 'AMT_CREDIT_SUM_OVERDUE_MEAN': 0, 'AMT_ANNUITY_BUREAU_ACTIVE': 0, 'CURRENCY_CNT': 0, 'DAYS_MOST_RECENT_CREDIT': 0, 'CREDIT_PROLONG_COUNT': 0, 'BURO_APP_CNT': 0}, inplace=True)
application_all_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
application_all_proc[bureau_stat_agg_features] = application_all_proc[bureau_stat_agg_features].replace([np.nan], 0)
application_all_proc.drop_duplicates(inplace=True)
application_all_proc = application_all_proc.sort_values(by=['SK_ID_CURR']).reset_index(drop=True)

del bureau_stat_agg_features
gc.collect()

score, predictions = lightgbm_predict(application_all_proc, categorical_features, rounds=50, n_folds=2)

print(predictions.head())

# previous_application
print('previous_application')

previous_application = reduce_memory(pd.read_csv('./datasets/previous_application.csv').rename(index=str, columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE_PREV', 'AMT_ANNUITY': 'AMT_ANNUITY_PREV', 'AMT_CREDIT': 'AMT_CREDIT_PREV', 'AMT_GOODS_PRICE': 'AMT_GOODS_PRICE_PREV', 'AMT_DOWN_PAYMENT': 'AMT_DOWN_PAYMENT_PREV'}))

previous_application.replace(365243, np.nan, inplace= True) 

proc_df = previous_application.groupby(['SK_ID_CURR'])['SK_ID_PREV'].agg('count').reset_index(name='PREV_APP_CNT')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_PREV_APPROUVED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_PREV_REFUSED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Canceled'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_PREV_CANCELED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Unused offer'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_PREV_UNUSED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_PORTFOLIO'] == 'Cash'].groupby(['SK_ID_CURR'])['NAME_PORTFOLIO'].agg('count').reset_index(name='CNT_PREV_CASH_PORTFOLIO')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_PORTFOLIO'] == 'POS'].groupby(['SK_ID_CURR'])['NAME_PORTFOLIO'].agg('count').reset_index(name='CNT_PREV_POS_PORTFOLIO')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_PORTFOLIO'] == 'XNA'].groupby(['SK_ID_CURR'])['NAME_PORTFOLIO'].agg('count').reset_index(name='CNT_PREV_XNA_PORTFOLIO')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_PORTFOLIO'] == 'Cards'].groupby(['SK_ID_CURR'])['NAME_PORTFOLIO'].agg('count').reset_index(name='CNT_PREV_CARDS_PORTFOLIO')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_TYPE_PREV'] == 'Cash loans'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_TYPE_PREV'].agg('count').reset_index(name='CNT_PREV_CASH_LOANS')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_TYPE_PREV'] == 'Consumer loans'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_TYPE_PREV'].agg('count').reset_index(name='CNT_PREV_CONSUMER_LOANS')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CONTRACT_TYPE_PREV'] == 'Revolving loans'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_TYPE_PREV'].agg('count').reset_index(name='CNT_PREV_REVOLV_LOANS')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_YIELD_GROUP'] == 'XNA'].groupby(['SK_ID_CURR'])['NAME_YIELD_GROUP'].agg('count').reset_index(name='CNT_PREV_XNA_YIELD')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_YIELD_GROUP'] == 'middle'].groupby(['SK_ID_CURR'])['NAME_YIELD_GROUP'].agg('count').reset_index(name='CNT_PREV_MIDDLE_YIELD')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_YIELD_GROUP'] == 'high'].groupby(['SK_ID_CURR'])['NAME_YIELD_GROUP'].agg('count').reset_index(name='CNT_PREV_HIGH_YIELD')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_YIELD_GROUP'] == 'low_normal'].groupby(['SK_ID_CURR'])['NAME_YIELD_GROUP'].agg('count').reset_index(name='CNT_PREV_LOW_NORMAL_YIELD')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_YIELD_GROUP'] == 'low_action'].groupby(['SK_ID_CURR'])['NAME_YIELD_GROUP'].agg('count').reset_index(name='CNT_PREV_LOW_ACTION_YIELD')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = previous_application[previous_application['NAME_CLIENT_TYPE'] == 'Refreshed'].groupby(['SK_ID_CURR'])['NAME_CLIENT_TYPE'].agg('count').reset_index(name='CNT_PREV_CLIENT_TYPE_REFRESHED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')


previous_application['ASK_GRANTED_RATIO'] = previous_application['AMT_APPLICATION']/previous_application['AMT_CREDIT_PREV']
previous_application['ASK_GRANTED_DIFF'] = previous_application['AMT_APPLICATION']-previous_application['AMT_CREDIT_PREV']
previous_application['YEARS_TO_PAY'] = previous_application['AMT_ANNUITY_PREV']/previous_application['AMT_CREDIT_PREV']
previous_application['CREDIT_TO_GOODS_RATIO_PREV'] = previous_application['AMT_CREDIT_PREV']/previous_application['AMT_GOODS_PRICE_PREV']


prev_cols_agg = {
    'AMT_ANNUITY_PREV': ['min', 'max', 'mean'],
    'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
    'AMT_CREDIT_PREV': ['min', 'max', 'mean', 'sum'],
    'AMT_DOWN_PAYMENT_PREV': ['min', 'max', 'mean'],
    'AMT_GOODS_PRICE_PREV': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'RATE_INTEREST_PRIMARY': ['min', 'max'],      
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'SELLERPLACE_AREA': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['min', 'max', 'mean'],
    'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
    'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE': ['min', 'max', 'mean'],
    'DAYS_TERMINATION': ['min', 'max', 'mean'],
    'NFLAG_INSURED_ON_APPROVAL': ['mean', 'std'],
    'ASK_GRANTED_RATIO': ['min', 'max', 'mean', 'std'],
    'ASK_GRANTED_DIFF': ['min', 'max', 'mean', 'std'],
    'YEARS_TO_PAY': ['min', 'max', 'mean', 'std', 'sum'],
    'CREDIT_TO_GOODS_RATIO_PREV': ['min', 'max', 'mean', 'std']
}


previous_application, prev_stat_features = apply_agg(previous_application, prev_cols_agg, 'SK_ID_CURR', 'PREV')
application_all_proc = reduce_memory(pd.merge(application_all_proc, previous_application[['SK_ID_CURR']+prev_stat_features], on='SK_ID_CURR', how='left'))
application_all_proc.drop_duplicates(inplace=True)
application_all_proc = application_all_proc.sort_values(by=['SK_ID_CURR']).reset_index(drop=True)

application_all_proc.fillna({
    'PREV_APP_CNT': 0,
    'CNT_PREV_APPROUVED': 0,
    'CNT_PREV_REFUSED': 0,
    'CNT_PREV_CANCELED': 0,
    'CNT_PREV_UNUSED': 0,
    'CNT_PREV_POS_PORTFOLIO': 0,
    'CNT_PREV_CASH_PORTFOLIO': 0,
    'CNT_PREV_CARDS_PORTFOLIO': 0,
    'CNT_PREV_CASH_LOANS': 0,
    'CNT_PREV_CONSUMER_LOANS': 0,
    'CNT_PREV_REVOLV_LOANS': 0,
    'CNT_PREV_XNA_YIELD': 0,
    'CNT_PREV_MIDDLE_YIELD': 0,
    'CNT_PREV_HIGH_YIELD': 0,
    'CNT_PREV_LOW_NORMAL_YIELD': 0,
    'CNT_PREV_LOW_ACTION_YIELD': 0,
    'CNT_PREV_CLIENT_TYPE_REFRESHED': 0}, inplace=True)
application_all_proc.replace([np.inf, -np.inf], np.nan, inplace=True)


del previous_application, proc_df
gc.collect()


# credit_card_balance
print('credit_card_balance')

credit_card_balance = reduce_memory(pd.read_csv('./datasets/credit_card_balance.csv'))

proc_df = credit_card_balance[credit_card_balance['NAME_CONTRACT_STATUS']=='Active'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_CARD_ACTIVE')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')


proc_df = credit_card_balance[credit_card_balance['NAME_CONTRACT_STATUS']=='Completed'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_CARD_COMPLETED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

credit_card_balance['PAST_DUE'] = credit_card_balance['SK_DPD'].apply(lambda s : 1 if s>0 else 0)
credit_card_balance['PAST_DUE_TOL'] = credit_card_balance['SK_DPD_DEF'].apply(lambda s : 1 if s>0 else 0)

application_all_proc['CNT_CARD_ACTIVE_COMP_RATIO'] = application_all_proc['CNT_CARD_COMPLETED']/application_all_proc['CNT_CARD_ACTIVE']
credit_card_balance['AMT_BALANCE_LIMIT_RATIO'] = credit_card_balance['AMT_BALANCE']/credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
credit_card_balance['AMT_DRAWING_LIMIT_RATIO'] = credit_card_balance['AMT_DRAWINGS_CURRENT']/credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
credit_card_balance['AMT_PAYMENT_MIN_INST_RATIO'] = credit_card_balance['AMT_PAYMENT_CURRENT']/credit_card_balance['AMT_INST_MIN_REGULARITY']
credit_card_balance['AMT_RECEIVABLE_PRINCIPAL_RATIO'] = credit_card_balance['AMT_TOTAL_RECEIVABLE']/credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']


card_cols_agg = {
    'AMT_BALANCE_LIMIT_RATIO': ['min', 'max', 'mean', 'std'],
    'AMT_DRAWING_LIMIT_RATIO': ['max', 'mean', 'std'],
    'AMT_PAYMENT_MIN_INST_RATIO': ['min', 'max', 'mean', 'std'],
    'AMT_RECEIVABLE_PRINCIPAL_RATIO': ['min', 'max', 'mean', 'std'],
    'PAST_DUE': ['sum'],
    'PAST_DUE_TOL': ['sum'],
    'AMT_BALANCE': ['min', 'max', 'mean', 'std'],                
    'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'std'],
    'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'std'],      
    'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'std'],
    'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'std'],
    'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'std'],
    'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'std'],
    'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'std'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'std'],
    'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'std'],
    'AMT_RECIVABLE': ['min', 'max', 'mean', 'std'],
    'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'std'],
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'std'],
    'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'std'],
    'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'std'],
    'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'std'],
    'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean', 'std']
}

credit_card_balance, card_stat_features = apply_agg(credit_card_balance, card_cols_agg, 'SK_ID_CURR', 'CARD')

credit_card_balance = reduce_memory(credit_card_balance[['SK_ID_CURR']+card_stat_features])
credit_card_balance.drop_duplicates(inplace=True)

application_all_proc = reduce_memory(pd.merge(application_all_proc, credit_card_balance, on='SK_ID_CURR', how='left'))
application_all_proc[card_stat_features] = application_all_proc[card_stat_features].fillna(0) 

del credit_card_balance, card_stat_features
gc.collect()


application_all_proc.fillna({
    'CNT_CARD_ACTIVE_COMP_RATIO': 0,
    'CNT_CARD_ACTIVE': 0,
    'CNT_CARD_COMPLETED': 0
}, inplace=True)
application_all_proc.replace([np.inf, -np.inf], np.nan, inplace=True)


# installments_payments
print('installments_payments')

installments_payments = reduce_memory(pd.read_csv('./datasets/installments_payments.csv'))

installments_payments['DAYS_INSTA_PAY_RATIO'] = installments_payments['DAYS_INSTALMENT']/installments_payments['DAYS_ENTRY_PAYMENT']
installments_payments['AMT_INSTAL_PAY_RATIO'] = installments_payments['AMT_INSTALMENT']/installments_payments['AMT_PAYMENT']
installments_payments['LATE_PAYMENT'] = installments_payments['DAYS_INSTA_PAY_RATIO'].apply(lambda x : 1 if x > 1 else 0)
installments_payments['EARLY_PAYMENT'] = installments_payments['DAYS_INSTA_PAY_RATIO'].apply(lambda x : 1 if x < 1 else 0)
installments_payments['ONTIME_PAYMENT'] = installments_payments['DAYS_INSTA_PAY_RATIO'].apply(lambda x : 1 if x == 1 else 0)
installments_payments['UNDER_PAYMENT'] = installments_payments['AMT_INSTAL_PAY_RATIO'].apply(lambda x : 1 if x > 1 else 0)
installments_payments['OVER_PAYMENT'] = installments_payments['AMT_INSTAL_PAY_RATIO'].apply(lambda x : 1 if x <= 1 else 0)
installments_payments['PERCENT_NOT_PAYED'] = (installments_payments['AMT_INSTALMENT']-installments_payments['AMT_PAYMENT'])/installments_payments['AMT_INSTALMENT']


inst_cols_agg = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'NUM_INSTALMENT_NUMBER': ['sum', 'max', 'mean', 'std'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_INSTA_PAY_RATIO': ['min', 'max', 'mean', 'std'],
    'AMT_INSTAL_PAY_RATIO': ['min', 'max', 'mean', 'std'],
    'LATE_PAYMENT': ['mean', 'sum'],
    'EARLY_PAYMENT': ['mean', 'sum'],
    'UNDER_PAYMENT': ['max', 'mean', 'sum'],
    'OVER_PAYMENT': ['mean', 'sum'],
    'PERCENT_NOT_PAYED': ['min', 'max', 'mean', 'std']
}
installments_payments = reduce_memory(installments_payments)

installments_payments, inst_stat_features = apply_agg(installments_payments, inst_cols_agg, 'SK_ID_CURR', 'INST')

installments_payments = reduce_memory(installments_payments[['SK_ID_CURR']+inst_stat_features])
installments_payments.drop_duplicates(inplace=True)

application_all_proc = reduce_memory(pd.merge(application_all_proc, installments_payments, on='SK_ID_CURR', how='left'))

del installments_payments
gc.collect()

application_all_proc[inst_stat_features].fillna(0, inplace=True)
application_all_proc.replace([np.inf, -np.inf], np.nan, inplace=True)


# pos_balance
print('pos_balance')

pos_balance = reduce_memory(pd.read_csv('./datasets/POS_CASH_balance.csv'))

pos_balance['PAST_DUE'] = pos_balance['SK_DPD'].apply(lambda s : 1 if s>0 else 0)
pos_balance['PAST_DUE_TOL'] = pos_balance['SK_DPD_DEF'].apply(lambda s : 1 if s>0 else 0)
pos_balance['CNT_INSTAL_BY_MONTH'] = -pos_balance['CNT_INSTALMENT']/pos_balance['MONTHS_BALANCE']
pos_balance['CNT_INSTAL_FUTURE_BY_MONTH'] = -pos_balance['CNT_INSTALMENT_FUTURE']/pos_balance['MONTHS_BALANCE']
pos_balance['CNT_INSTAL_FUTURE_RATIO'] = pos_balance['CNT_INSTALMENT']/pos_balance['CNT_INSTALMENT_FUTURE']
pos_balance['CNT_INSTAL_FUTURE_MONTH_RATIO'] = pos_balance['CNT_INSTAL_BY_MONTH']/pos_balance['CNT_INSTALMENT_FUTURE']
pos_balance['CNT_INSTAL_FUTURE_MONTH_DIF'] = pos_balance['CNT_INSTAL_BY_MONTH']-pos_balance['CNT_INSTAL_FUTURE_BY_MONTH']

proc_df = pos_balance[pos_balance['NAME_CONTRACT_STATUS']=='Active'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_POS_ACTIVE')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = pos_balance[pos_balance['NAME_CONTRACT_STATUS']=='Completed'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_POS_COMPLETED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = pos_balance[pos_balance['NAME_CONTRACT_STATUS']=='Signed'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_POS_SIGNED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')

proc_df = pos_balance[pos_balance['NAME_CONTRACT_STATUS']=='Returned to the store'].groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg('count').reset_index(name='CNT_POS_RETURNED')
application_all_proc = pd.merge(application_all_proc, proc_df, on='SK_ID_CURR', how='left')


pos_cols_agg = {
    'CNT_INSTAL_BY_MONTH': ['min', 'max', 'mean', 'std'],
    'CNT_INSTAL_FUTURE_BY_MONTH': ['min', 'max', 'mean', 'std'],
    'CNT_INSTAL_FUTURE_RATIO': ['min', 'max', 'mean', 'std'],
    'CNT_INSTAL_FUTURE_MONTH_RATIO': ['min', 'max', 'mean', 'std'],
    'CNT_INSTAL_FUTURE_MONTH_DIF': ['min', 'max', 'mean', 'std'],
    'CNT_INSTALMENT': ['min', 'max', 'mean', 'std'],
    'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'std'],
    'PAST_DUE': ['max', 'sum'],
    'PAST_DUE_TOL': ['max', 'sum']
}
pos_balance, pos_stat_features = apply_agg(pos_balance, pos_cols_agg, 'SK_ID_CURR', 'POS')

pos_balance = pos_balance[['SK_ID_CURR']+pos_stat_features]
pos_balance.drop_duplicates(inplace=True)

application_all_proc = reduce_memory(pd.merge(application_all_proc, pos_balance, on='SK_ID_CURR', how='left'))

del pos_balance
gc.collect()

pos_features = [
    'CNT_POS_ACTIVE',
    'CNT_POS_COMPLETED',
    'CNT_POS_SIGNED',
    'CNT_POS_RETURNED'] + pos_stat_features

application_all_proc[pos_features].fillna(0, inplace=True)
application_all_proc.replace([np.inf, -np.inf], np.nan, inplace=True)


score, predictions = lightgbm_predict(application_all_proc, categorical_features)

print(predictions)

predictions.to_csv('credit_submit.csv', encoding='utf-8', index=False)














