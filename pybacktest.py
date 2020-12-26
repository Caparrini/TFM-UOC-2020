import logging
import os
import deap
import xgboost
import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from mloptimizer.genoptimizer import TreeOptimizer, XGBClassifierOptimizer, ForestOptimizer
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import joblib
import plotly
import xgboost as xgb
from copy import deepcopy
import argparse
import boto3
import io
import logging

LOG_PATH=""
def init_logger(filename='optimization.log'):
    logging_params = {
        # 'stream': sys.stdout,
        'level': logging.DEBUG,
        'format': '%(asctime)s:%(levelname)s:[L%(lineno)d]:%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'filename': os.path.join(LOG_PATH, filename)
    }
    print(os.listdir("."))
    logging.basicConfig(**logging_params)
    logging.debug('Logger configured')

# Read the factor data
def get_dataframe(file, index_col, aws=False):
    if aws:
        s3 = boto3.client('s3')
        myob = s3.get_object(Bucket="tfm-portfolio-opt", Key=file)
        df = pd.read_csv(io.BytesIO(myob['Body'].read()))
        if index_col is not None:
            df = df.pivot_table(index=list(df.columns[index_col].values))
    else:
        df = pd.read_csv(file, index_col=index_col)
    return df

def get_pricing(file="data/SHARADAR_SEP_5659e77b6ffc1f36d023b6a3987da31c.csv"):
    my_data = get_dataframe(file)
    columns = ['date','ticker','close']
    my_data = my_data[my_data['ticker'].isin(tickers)][columns]

# Selected factors and target var
def columns_target():
    selected_factors = ['momentum_6m', 'momentum_11m', 'roe', 'beta_3Y', 
                        'beta_3Y_coef', 'price-to-book', 'earnings-to-price', 
                        'price-to-sales', 'marketcap', 'enterprise-value', 
                        'evebit', 'evebitda', 'returns_12m_lagged_12m', 'returns_12m_lagged_24m', 
                        'operating cashflow-to-price', 'investment-to-price', 'earnings-per-share', 
                        'current-ratio', 'operating cashflow-to-equity', 'capex']
    target_var = ['target']
    return selected_factors, target_var

#SP500 Daily Benchmark
class SP500:
    
    def __init__(self, aws=False):
        SP500_daily_data = get_dataframe("data/SP500.csv", index_col=None, aws=aws)
        SP500_daily_data['Date'] = pd.to_datetime(SP500_daily_data['Date'])
        SP500_daily_data = SP500_daily_data.pivot_table(index='Date', values='Adj Close')
        SP500_daily_data.rename(columns = {'Adj Close': 'Value'}, inplace = True)
        self.daily_data = SP500_daily_data
        
    def dates_contained(self, dates):
        return any([a in self.daily_data.index for a in dates])

    def get_returns(self, dates):
        previous_date = dates[0]
        sp500_carried_return = 1
        self.returns = [sp500_carried_return]
        for d in dates[1:]:
            #print("Calculate SP500 returns on {}".format(d))
            iter_returns = self.daily_data.loc[d]/self.daily_data.loc[previous_date]
            sp500_carried_return *= iter_returns
            #print("Carried return sp500 {}".format(sp500_carried_return))
            self.returns.append(float(sp500_carried_return))
            previous_date = d
        return self.returns
        
    def gen_returns(self, dates):
        if self.dates_contained(dates):
            return self.get_returns(dates)
        else:
            print("Some date does not exist on SP500_daily_data")

# Las sumas absolutas de fechas pueden dar como resultado un día ilegal que cambiaremos por el siguiente disponible
def get_index_dates(df):
    index_dates = df.index.get_level_values(level='date').drop_duplicates()
    return index_dates

def legal_date(date, legal_dates, direction=1):
    while date not in legal_dates:
        date = date + direction*relativedelta(days=1)
        if date > legal_dates[-1]:
            return -1
    return date

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month



# Backtest Object
class Backtest:
    def create_target_var_quant(self, df):
        my_df = df.copy()
        def custom_target(x):
            value = 0
            
        #my_df['new_target'] = np.where(
        #    my_df['target']>=my_df['target'].quantile(0.5), 1, 0)
        my_df['new_target'] = pd.qcut(my_df['target'], 5, labels=range(5), precision=5)
        return my_df['new_target']
    
    def create_target_var(self, df):
        my_df = df.copy()
        def custom_target(x):
            if x >= 0.15:
                return 4
            elif 0.05 <= x and x < 0.15:
                return 3
            elif 0 <= x and x < 0.05:
                return 2
            elif -0.15 <= x and x < 0:
                return 1
            else:
                return 0
            
        #my_df['new_target'] = np.where(
        #    my_df['target']>=my_df['target'].quantile(0.5), 1, 0)
        my_df['new_target'] = my_df['target'].apply(custom_target)
        return my_df['new_target']
    
    def get_optimized_clf(self, X, y):
        opt = ForestOptimizer(X, y, 'log')
        clf = opt.optimize_clf(30,5)
        return clf

    def get_pre_xgb(self):
        return xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=11, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.048, max_delta_step=0, max_depth=22,
              min_child_weight=1, missing=None,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=229, n_jobs=-1, nthread=-1, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, seed=0, subsample=0.711,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

    def get_pre_rf(Self):
        return RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                       criterion='gini', max_depth=9, max_features=0.79,
                       max_leaf_nodes=None, max_samples=0.3,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0, n_estimators=119, n_jobs=-1,
                       oob_score=True, random_state=None, verbose=0,
                       warm_start=False)

    def get_clf_try1(self):
        return DecisionTreeClassifier(ccp_alpha=0.00013, class_weight='balanced',
                       criterion='gini', max_depth=30, max_features=None,
                       max_leaf_nodes=None, min_impurity_decrease=0.0002,
                       min_impurity_split=None, min_samples_leaf=9,
                       min_samples_split=47, min_weight_fraction_leaf=0.0,
                       presort='deprecated', random_state=None,
                       splitter='best')

        
    def get_clf(self, x_train, y_train, x_test, y_test):
        #clf = self.get_clf_try1()
        #clf = self.get_optimized_clf(x_train, y_train)
        clf = self.get_pre_rf()
        #clf = self.get_pre_xgb()
        #clf = xgb.XGBClassifier()
        print(clf)
        clf.fit(x_train, y_train)
        clf2 = deepcopy(clf)
        logging.info("Classifier: {}".format(clf))
        logging.info("Feature importances: {}".format(clf.feature_importances_))
        predict = clf2.predict(x_test)
        acc = accuracy_score(predict, y_test)
        bacc = balanced_accuracy_score(predict, y_test)
        logging.info("Accuracy: {}".format(acc))
        logging.info("Balance accuracy: {}".format(bacc))
        self.clf = clf
        return clf, acc, bacc
    
    def legal_date(self, date, direction=1):
        while date not in self.index_dates:
            date = date + direction*relativedelta(days=1)
            if date > self.index_dates[-1]:
                logging.info("Last period")
                return self.index_dates[-1]
        return date

    def legal_pricing_date(self, date, direction=1):
        pricing_index_dates = self.pricing.index.get_level_values(level='date').drop_duplicates()
        while date not in pricing_index_dates:
            date = date + direction*relativedelta(days=1)
            if date > pricing_index_dates[-1]:
                logging.info("Last period")
                return pricing_index_dates[-1]
        return date
    
    def __init__(self, factor_data, pricing, sp500_comp, aws=False):
        self.aws = aws
        self.factor_data = factor_data
        self.pricing = pricing
        self.sp500_comp = sp500_comp
        self.sp500_dates = sp500_comp['date'].drop_duplicates()
        self.index_dates = self.factor_data.index.get_level_values(level='date').drop_duplicates()
        self.rebalancing_dates = []
        self.returns = []
        self.returns_long = []
        self.returns_short = []
        self.current_date = pricing.index.get_level_values(level='date').max()
        self.selected_factors, self.target_var = columns_target()
        self.SP500_pricing = SP500(self.aws)
        self.set_deltas()

    def set_deltas(self, years_training=3, months_rebalance=6, year_danger=1):
        #S
        self.size_training_period = relativedelta(years=years_training)
        #H
        self.rebalancing_period = relativedelta(months=months_rebalance)
        self.months_rebalance = months_rebalance
        #L
        self.danger_period = relativedelta(years=year_danger)
        
    def init_dates(self):
        #Instante inicial (primer dia de datos + 3 años)
        self.tzero = self.legal_date(self.index_dates[0] + 
                                self.size_training_period + self.danger_period)
        self.t_ini_x = self.legal_date(self.tzero - self.danger_period - self.size_training_period)
        #t_fin_x no incluida
        self.t_fin_x = self.legal_date(self.t_ini_x + self.size_training_period)
        self.t_fin_x_previous = self.legal_date(self.t_fin_x - relativedelta(days=1), -1)
        self.tzero_previous = self.legal_date(self.tzero - relativedelta(days=1), -1)

    def set_date(self, custom_rebalance_date):
        #Instante inicial (primer dia de datos + 3 años)
        self.tzero = custom_rebalance_date
        self.t_ini_x = self.legal_date(self.tzero - self.danger_period - self.size_training_period)
        #t_fin_x no incluida
        self.t_fin_x = self.legal_date(self.t_ini_x + self.size_training_period)
        self.t_fin_x_previous = self.legal_date(self.t_fin_x - relativedelta(days=1), -1)
        self.tzero_previous = self.legal_date(self.tzero - relativedelta(days=1), -1)
        
    def update_dates(self):
        #Instante inicial (primer dia de datos + 3 años)
        self.tzero = self.legal_date(self.tzero + self.rebalancing_period)
        self.t_ini_x = self.legal_date(self.tzero - self.danger_period - self.size_training_period)
        #t_fin_x no incluida
        self.t_fin_x = self.legal_date(self.t_ini_x + self.size_training_period)
        self.t_fin_x_previous = self.legal_date(self.t_fin_x - relativedelta(days=1), -1)
        self.tzero_previous = self.legal_date(self.tzero - relativedelta(days=1), -1)
        
    def print_info(self):
        logging.info("Dataset para entrenamiento en el rango ({},{}) con un tamaño {}".format(
        self.t_ini_x.strftime('%Y-%m-%d'),
        self.t_fin_x_previous.strftime('%Y-%m-%d'),
        self.factor_data.loc[self.t_ini_x:self.t_fin_x_previous].shape
        ))
        logging.info("Dataset para test en el rango ({},{}) con un tamaño {}".format(
            self.t_fin_x.strftime('%Y-%m-%d'),
            self.tzero_previous.strftime('%Y-%m-%d'),
            self.factor_data.loc[self.t_fin_x:self.tzero_previous].shape
        ))
        logging.info("Fecha para la generación del portfolio: {}".format(self.tzero.strftime('%Y-%m-%d')))
        print("Fecha para la generación del portfolio: {}".format(self.tzero.strftime('%Y-%m-%d')))

    
    def get_current_sp500_components(self):
        self.current_sp500 = []
        current = []
        added = []
        historical = []
        removed = []
        current_date = self.tzero
        while len(self.current_sp500) < 500:
            last_sp500_comp_date = self.sp500_dates[self.sp500_dates<=current_date].iloc[0]
            logging.info("Latest date with sp500 event {}".format(last_sp500_comp_date))
            current = self.sp500_comp[(self.sp500_comp['date']==last_sp500_comp_date) 
                                                      & (self.sp500_comp['action']=='current')
                                                        ]['ticker'].tolist()
            added = self.sp500_comp[(self.sp500_comp['date']==last_sp500_comp_date) 
                                                      & (self.sp500_comp['action']=='added')
                                                        ]['ticker'].tolist()
            historical = self.sp500_comp[(self.sp500_comp['date']==last_sp500_comp_date) 
                                                      & (self.sp500_comp['action']=='historical')
                                                        ]['ticker'].tolist()
            removed.extend(self.sp500_comp[(self.sp500_comp['date']==last_sp500_comp_date) 
                                                      & (self.sp500_comp['action']=='removed')
                                                        ]['ticker'].tolist())
            sp_set = set(current) | set(historical) | set(added)
            sp_set = sp_set - set(removed)
            sp_set = sp_set | set(self.current_sp500)
            self.current_sp500.extend(list(sp_set))
            current_date = last_sp500_comp_date - relativedelta(days=1)
        logging.info("Current SP500 tickers length {}".format(len(self.current_sp500)))

    def generate_train_test(self):
        # Filtering the SP500 tickers for the date
        logging.info("Filtering dataset to the SP500 of the date {}".format(self.tzero))
        self.get_current_sp500_components()
        
        self.iter_sp500_factor_data = self.factor_data[
            self.factor_data.index.get_level_values('asset').isin(self.current_sp500)]
        
        # Generate dataframes train-val and features for the rebalance date
        logging.info("Generar train")
        iter_train_factors = self.iter_sp500_factor_data.loc[self.t_ini_x:self.t_fin_x_previous].copy()
        iter_train_factors['target'] = self.create_target_var(iter_train_factors)
        features_train = iter_train_factors[self.selected_factors]
        target_train = iter_train_factors[self.target_var]
        logging.info("Tamaño filtrando componentes del SP500 del conjunto train es {}".format(
        features_train.shape))
                
        logging.info("Generar test")
        iter_test_factors = self.iter_sp500_factor_data.loc[self.t_fin_x:self.tzero_previous].copy()
        iter_test_factors['target'] = self.create_target_var(iter_test_factors)
        features_test = iter_test_factors[self.selected_factors]
        target_test = iter_test_factors[self.target_var]
        logging.info("Tamaño filtrando componentes del SP500 del conjunto test es {}".format(
        features_test.shape))
        return features_train, target_train.values.ravel(), features_test, target_test.values.ravel()
    
    def generate_ranking(self):
        logging.info("Rebalance portfolio on {}".format(self.tzero.strftime('%Y-%m-%d')))
        
        if len(self.rebalancing_dates) == 0:
            logging.info("First portfolio: Iniciar returns con 1")
            self.returns.append(1)
            self.returns_long.append(1)
            self.returns_short.append(1)
        self.rebalancing_dates.append(self.tzero)
        factors_rebalance_date = self.factor_data[
            self.factor_data.index.get_level_values('asset').isin(self.current_sp500)
        ].loc[self.tzero]
        stocks = factors_rebalance_date.index.tolist()
        predictions = self.clf.predict_proba(factors_rebalance_date[self.selected_factors])
        # Rank quantile 0 (worst performing)
        rank_worst = predictions[:, 0].tolist()
        # Rank quantile 4 (best performing)
        rank_best = predictions[:, 4].tolist()
        d_long = {'stocks': stocks, 'ranks': rank_best}
        d_short = {'stocks': stocks, 'ranks': rank_worst}
        ranking_long = pd.DataFrame(d_long)
        ranking_long.sort_values(by=['ranks'], ascending=False, inplace=True)
        
        ranking_short = pd.DataFrame(d_short)
        ranking_short.sort_values(by=['ranks'], ascending=False, inplace=True)
        logging.info("Ranking short")
        logging.info(ranking_short)
        logging.info("Ranking long")
        logging.info(ranking_long)
        return ranking_long, ranking_short
    
    def generate_rebalanced_portfolio(self, ranking_long, ranking_short):
        short_n = 230
        long_n = 230
        long_portfolio = pd.DataFrame({'stocks':ranking_long['stocks'][:long_n], 'weights':1./long_n})
        short_portfolio = pd.DataFrame({'stocks':ranking_short['stocks'][:short_n], 'weights':-1./short_n})
        portfolio = pd.concat([long_portfolio, short_portfolio]).reset_index()
        del portfolio['index']
        logging.info(portfolio)
        return portfolio
    
    def plot_vs_market(self, market):
        
        df = pd.DataFrame(np.array(self.rebalancing_dates), columns=['date'])
        df['date'] = df['date'].dt.date
        #df['Strategy'] = self.returns
        df['Strategy_long returns'] = self.returns_long
        #df['Strategy_short'] = self.returns_short
        df['SP500 returns'] = market.gen_returns(self.rebalancing_dates)
        #df = px.data.stocks()
        fig = px.line(df, x='date', y=df.columns,
                      hover_data={"date": "|%B %d, %Y"},
                      width=800, height=400)
        fig.update_layout(title_text='Returns', title_x=0.5)
        fig.update_xaxes(
            dtick="M10",
            tickformat="%Y %b")
        fig.show()
        try:
            plotly.offline.plot(fig, filename=os.path.join(self.rebalance_folder, "returns.html"))
            df.to_csv(os.path.join(self.rebalance_folder, "returns.csv"))
        except:
            print("Cannot save plotly")

    def create_rebalance_folder(self, date):
        self.rebalance_folder = os.path.join(self.backtest_folder, date)
        if not os.path.exists(self.rebalance_folder):
            logging.info("Make folder {} for rebalance date".format(self.rebalance_folder))
            os.makedirs(self.rebalance_folder)
        else:
            logging.error("FOLDER {} SHOULD NOT EXIST".format(self.rebalance_folder))
            
    def save_files(self, portfolio, clf, importances, accuracies, aws=False):
        aws = self.aws
        logging.info("Saving files of the rebalance date {}".format(
        self.tzero.strftime('%Y-%m-%d')))
        portfolio_file = os.path.join(self.rebalance_folder, "portfolio.csv")
        portfolio.to_csv(portfolio_file)
        clf_file = os.path.join(self.rebalance_folder, "clf.joblib")
        joblib.dump(clf, clf_file)
        importances_file = os.path.join(self.rebalance_folder, "importances.csv")
        importances.to_csv(importances_file)
        accuracies_file = os.path.join(self.rebalance_folder, "accuracies.csv")
        accuracies.to_csv(accuracies_file)
        if aws:
            s3 = boto3.client('s3')
            bucket = "tfm-portfolio-opt"
            response = s3.upload_file(portfolio_file, bucket, portfolio_file)
            response = s3.upload_file(clf_file, bucket, clf_file)
            response = s3.upload_file(importances_file, bucket, importances_file)
            response = s3.upload_file(accuracies_file, bucket, accuracies_file)

    def update_returns(self, portfolio, i):
        iter_returns = self.pricing[portfolio['stocks']].loc[self.tzero]/self.pricing[portfolio['stocks']].loc[self.rebalancing_dates[i]]-1
        portfolio['returns_iter']=np.multiply(portfolio['weights'], iter_returns.values)
        returns = portfolio['returns_iter'].sum()+1
        returns_long = portfolio[portfolio['weights']>0]['returns_iter'].sum()+1
        returns_short = portfolio[portfolio['weights']<0]['returns_iter'].sum()+1
        logging.info("Retorno del último periodo: {}".format(returns))
        logging.info("Retorno del último periodo (long): {}".format(returns_long))
        logging.info("Retorno del último periodo (short): {}".format(returns_short))
        carried_return = self.returns[i]*returns
        carried_return_long = self.returns_long[i]*returns_long
        carried_return_short = self.returns_short[i]*returns_short
        self.returns.append(carried_return)
        self.returns_long.append(carried_return_long)
        self.returns_short.append(carried_return_short)
        logging.info("Retorno acumulado: {}".format(carried_return))
        logging.info("Retorno acumulado (long): {}".format(carried_return_long))
        logging.info("Retorno acumulado (short): {}".format(carried_return_short))


    def update_returns_merge(self, portfolio, i, top_n=0):
        if top_n > 0:
            portfolio = portfolio.loc[pd.np.r_[0:top_n, 230:230+top_n]]
            portfolio.loc[portfolio['weights']>0,'weights'] = 1./top_n
            portfolio.loc[portfolio['weights']<0,'weights'] = -1./top_n
        iter_returns = self.pricing[portfolio['stocks']].loc[self.rebalance_dates[i+1]]/self.pricing[portfolio['stocks']].loc[self.rebalance_dates[i]]-1
        portfolio['returns_iter']=np.multiply(portfolio['weights'], iter_returns.values)
        returns = portfolio['returns_iter'].sum()+1
        returns_long = portfolio[portfolio['weights']>0]['returns_iter'].sum()+1
        returns_short = portfolio[portfolio['weights']<0]['returns_iter'].sum()+1
        logging.info("Retorno del último periodo: {}".format(returns))
        logging.info("Retorno del último periodo (long): {}".format(returns_long))
        logging.info("Retorno del último periodo (short): {}".format(returns_short))
        carried_return = self.returns[i]*returns
        carried_return_long = self.returns_long[i]*returns_long
        carried_return_short = self.returns_short[i]*returns_short
        self.returns.append(carried_return)
        self.returns_long.append(carried_return_long)
        self.returns_short.append(carried_return_short)
        logging.info("Retorno acumulado: {}".format(carried_return))
        logging.info("Retorno acumulado (long): {}".format(carried_return_long))
        logging.info("Retorno acumulado (short): {}".format(carried_return_short))

    def generate_rebalance_dates(self, years_training=3, months_rebalance=6, year_danger=1):
        ## Compute all rebalance dates
        self.set_deltas(years_training, months_rebalance, year_danger)
        self.init_dates()

        self.rebalance_dates = [self.tzero]

        next_date = self.legal_date(self.tzero + self.rebalancing_period)

        while (next_date < self.index_dates[-1]) & (diff_month(self.index_dates[-1], self.tzero)>=self.months_rebalance):
            self.rebalance_dates.append(next_date)
            next_date = self.legal_date(next_date + self.rebalancing_period)

        self.rebalancing_dates = self.rebalance_dates

    def launch_job(self, d):
        self.set_date(d)
        self.print_info()
        self.create_rebalance_folder(self.tzero.strftime('%Y%m%d'))
        # Get clf
        features_train, target_train, features_test, target_test = self.generate_train_test()
        clf, acc, bacc = self.get_clf(features_train, target_train, features_test, target_test)
        # Rebalance portfolio
        iter_p_long, iter_p_short = self.generate_ranking()
        portfolio = self.generate_rebalanced_portfolio(iter_p_long, iter_p_short)
        importances = pd.DataFrame(data=self.clf.feature_importances_.reshape(1,20),
            columns=self.selected_factors,
            index=[self.tzero])
        accuracies = pd.DataFrame(data=np.array([acc, bacc]).reshape(1,2), columns=['acc', 'bacc'],
                                     index=[self.tzero])
        self.save_files(portfolio, clf, importances, accuracies)


    def launch_jobs(self):
        client = boto3.client('batch', region_name='eu-west-3')
        for d in self.rebalance_dates:
            #self.launch_job(d)
            #b2 = Backtest(self.factor_data, self.pricing, self.sp500_comp)
            #b2.backtest_folder = "backtest_20201122"
            #b2.launch_job(d)
            #os.system("python3 pybacktest.py executeRebalanceDate -d {} -f {}".format(
            #    d.strftime('%Y%m%d'), self.backtest_folder))
            command = "python3 /root/pybacktest.py executeRebalanceDate -d {} -f {}".format(
                d.strftime('%Y%m%d'), self.backtest_folder).split(" ")
            response = client.submit_job(
                           jobName='AutoJob-{}'.format(d.strftime('%Y%m%d')),
                           jobQueue='tfm-portfolioopt-jobs-queue',
                           jobDefinition='tfm-portfolioopt-jobdef:8',
                           containerOverrides={
                               'vcpus': 2,
                               'memory': 4096,
                               'command': command
                           },
                           tags={
                               'Grupo': 'TFM'
                           }
                       )

    def paralel_batch_run(self):
        # Initialize context
        self.backtest_folder = "backtest_"+datetime.datetime.now().strftime('%Y%m%d')
        if not os.path.exists(self.backtest_folder):
            logging.info("Make folder {} for backtest".format(self.backtest_folder))
            os.makedirs(self.backtest_folder)
        else:
            logging.error("FOLDER {} SHOULD NOT EXIST".format(self.backtest_folder))
        ## Compute all rebalance dates
        self.generate_rebalance_dates()

        
        #Launch Jobs
        ## Launch a Job for each rebalance date
        self.launch_jobs()

        # Merge results
        self.merge_results(self.backtest_folder)

    def merge_results(self, backtest_folder, top_n=0):
        self.returns = [1]
        self.returns_long = [1]
        self.returns_short = [1]
        self.rebalance_dates.append(
            self.legal_pricing_date(self.rebalance_dates[-1] + self.rebalancing_period)
        )
        for i in range(0, len(self.rebalance_dates)-1):
            current_folder = os.path.join(
                backtest_folder,
                self.rebalance_dates[i].strftime('%Y%m%d'))
            portfolio = get_dataframe(os.path.join(current_folder, "portfolio.csv"),
                                      index_col=[0])
            self.update_returns_merge(portfolio, i, top_n)

    def get_feature_importances(self, backtest_folder):
        self.rebalance_dates.append(
            self.legal_pricing_date(self.rebalance_dates[-1] + self.rebalancing_period)
        )
        df = None
        for i in range(0, len(self.rebalance_dates)-1):
            current_folder = os.path.join(
                backtest_folder,
                self.rebalance_dates[i].strftime('%Y%m%d'))
            feature_importances = get_dataframe(os.path.join(current_folder, "importances.csv"),
                                      index_col=[0])
            if df is None:
                df = feature_importances.copy()
            else:
                df = df.append(feature_importances)
        return df

    def get_accuracies(self, backtest_folder):
        self.rebalance_dates.append(
            self.legal_pricing_date(self.rebalance_dates[-1] + self.rebalancing_period)
        )
        df = None
        for i in range(0, len(self.rebalance_dates)-1):
            current_folder = os.path.join(
                backtest_folder,
                self.rebalance_dates[i].strftime('%Y%m%d'))
            feature_importances = get_dataframe(os.path.join(current_folder, "accuracies.csv"),
                                      index_col=[0])
            if df is None:
                df = feature_importances.copy()
            else:
                df = df.append(feature_importances)
        return df

    def run(self, years_training=3, months_rebalance=6, year_danger=1):
        
        self.backtest_folder = "backtest_"+datetime.datetime.now().strftime('%Y%m%d')
        if not os.path.exists(self.backtest_folder):
            logging.info("Make folder {} for backtest".format(self.backtest_folder))
            os.makedirs(self.backtest_folder)
        else:
            logging.error("FOLDER {} SHOULD NOT EXIST".format(self.backtest_folder))
        
        self.set_deltas(years_training, months_rebalance, year_danger)
        
        self.init_dates()
        
        self.print_info()
        
        self.create_rebalance_folder(self.tzero.strftime('%Y%m%d'))
        
        # Get clf
        features_train, target_train, features_test, target_test = self.generate_train_test()
        clf, acc, bacc = self.get_clf(features_train, target_train, features_test, target_test)
        
        # Rebalance portfolio
        iter_p_long, iter_p_short = self.generate_ranking()
        portfolio = self.generate_rebalanced_portfolio(iter_p_long, iter_p_short)
        
        importances = pd.DataFrame(data=self.clf.feature_importances_.reshape(1,20),
                columns=self.selected_factors,
                index=[self.tzero])
            
        accuracies = pd.DataFrame(data=np.array([acc, bacc]).reshape(1,2), columns=['acc', 'bacc'],
                                     index=[self.tzero])
            
        self.save_files(portfolio, clf, importances, accuracies)

        i = 0
        self.plot_vs_market(self.SP500_pricing)
        while (self.tzero < self.index_dates[-1]) & (diff_month(self.index_dates[-1], self.tzero)>=self.months_rebalance):
            self.update_dates()
            self.print_info()
            self.create_rebalance_folder(self.tzero.strftime('%Y%m%d'))
            # Get clf
            features_train, target_train, features_test, target_test = self.generate_train_test()
            clf, acc, bacc = self.get_clf(features_train, target_train, features_test, target_test)

            
            self.update_returns(portfolio, i)
            
            # Rebalance portfolio
            iter_p_long, iter_p_short = self.generate_ranking()
            portfolio = self.generate_rebalanced_portfolio(iter_p_long, iter_p_short)
            
            importances = pd.DataFrame(data=self.clf.feature_importances_.reshape(1,20),
                columns=self.selected_factors,
                index=[self.tzero])
            
            accuracies = pd.DataFrame(data=np.array([acc, bacc]).reshape(1,2), columns=['acc', 'bacc'],
                                     index=[self.tzero])
            

            self.save_files(portfolio, clf, importances, accuracies)
            self.plot_vs_market(self.SP500_pricing)
            i = i + 1
        
        logging.info("Checking returns on latest portfolio")
        self.rebalancing_dates.append(self.current_date)
        self.tzero = self.current_date
        self.create_rebalance_folder(self.tzero.strftime('%Y%m%d'))
        self.save_files(portfolio, clf, importances, accuracies)
        self.update_returns(portfolio, i)
        self.plot_vs_market(self.SP500_pricing)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Execute rebalance date")
    tasks = parser.add_subparsers(title="commands",
                                  description="available commands",
                                  dest="task",
                                  metavar="")
    exe_reb_date = tasks.add_parser("executeRebalanceDate", help="Executes a single rebalance date")
    exe_reb_date.add_argument("-d", "--rebalance_date", required=True,
                              help="Date for compute rebalance classifier and portfolio")
    exe_reb_date.add_argument("-f", "--backtest_folder", required=True,
                              help="Folder to store info")
    return parser.parse_args()

def main_run(aws=False, years_training=3, months_rebalance=6, year_danger=1):
    b = return_backtest(aws)
    b.run(years_training, months_rebalance, year_danger)

def return_backtest(aws=False):
    # Init logger
    init_logger()
    # Read factor data
    all_factors = get_dataframe("data/cleaned_factors.csv", index_col=[0,1], aws=aws)
    all_factors.index = all_factors.index.set_levels(pd.to_datetime(all_factors.index.levels[0]), level='date')
    tickers = all_factors.index.levels[1].tolist()
    print(all_factors.head())
    # Read pricing
    pricing = get_dataframe("data/pricing.csv", index_col=[0], aws=aws)
    pricing.index = pd.to_datetime(pricing.index)
    print(pricing.head())
    # Read SP500
    SP500_comp = get_dataframe("data/SHARADAR_SP500_587ff151b73b19580ab063e68c865d69.csv", None, aws=aws)
    SP500_comp['date'] = pd.to_datetime(SP500_comp['date'])
    print(SP500_comp.head())

    b = Backtest(all_factors, pricing, SP500_comp)
    return b

def job_run(aws=False):
    args = parse_arguments()

    if args.task == "executeRebalanceDate":
        reb_date = pd.to_datetime(args.rebalance_date)
        backtest_folder = args.backtest_folder

        logfile = "optimization_{}.log".format(args.rebalance_date)
        init_logger(logfile)
        # Read factor data
        aws=False
        all_factors = get_dataframe("data/cleaned_factors.csv", index_col=[0,1], aws=aws)
        all_factors.index = all_factors.index.set_levels(pd.to_datetime(all_factors.index.levels[0]), level='date')
        tickers = all_factors.index.levels[1].tolist()
        print(all_factors.head())
        # Read pricing
        pricing = get_dataframe("data/pricing.csv", index_col=[0], aws=aws)
        pricing.index = pd.to_datetime(pricing.index)
        print(pricing.head())
        # Read SP500
        SP500_comp = get_dataframe("data/SHARADAR_SP500_587ff151b73b19580ab063e68c865d69.csv", None, aws=aws)
        SP500_comp['date'] = pd.to_datetime(SP500_comp['date'])
        print(SP500_comp.head())
        backtest_job = Backtest(all_factors, pricing, SP500_comp)
        backtest_job.backtest_folder = backtest_folder
        backtest_job.launch_job(reb_date)

        #Copy log to aws
        if aws:
            s3 = boto3.client('s3')
            bucket = "tfm-portfolio-opt"
            log_aws = os.path.join(args.backtest_folder, logfile)
            response = s3.upload_file(logfile, bucket, log_aws)

if __name__=="__main__":
    #python3 pybacktest.py executeRebalanceDate -d 20051205 -f backtest_20201122
    job_run()
