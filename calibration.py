from argparse import Namespace
from SurvivalEVAL import QuantileRegEvaluator
import numpy as np
import pandas as pd
from datasets import generate_T_C
from training import get_quantile_time, surv_from_quantiles
from pycox.evaluation import EvalSurv
from MakeSurvivalCalibratedAgain.icp.scorer import QuantileRegressionNC
from MakeSurvivalCalibratedAgain.icp import ConformalSurvDist
from MakeSurvivalCalibratedAgain.utils.util_survival import xcal_from_hist

def csd_cov(target_alphas, setting, surv_model, df_train, df_cal, df_test, x_cal, x_test, durations_cal):
    predictors = csd(target_alphas, surv_model, df_train, df_cal, x_cal, x_test, durations_cal)
    
    if setting in range(1,7):
        T_test = df_test['T'].values
        coverage = (predictors <= T_test).mean(axis=1)
        return coverage, predictors.mean(axis=1)
    else:
        return predictors.mean(axis=1)

def csd_metrics_estimation(target_alphas, setting, surv_model, df_train, df_cal, df_test, x_cal, x_test, durations_cal):
    predictors = csd(target_alphas, surv_model, df_train, df_cal, x_cal, x_test, durations_cal)
    
    quant_preds = np.concatenate([np.zeros((predictors.T.shape[0], 1)), predictors.T], axis=1)
    quant_levels = np.concatenate([np.array([0]), target_alphas])

    evaler = QuantileRegEvaluator(quant_preds, quant_levels, df_test['duration'], df_test['event'], df_train['duration'], df_train['event'],
                                      predict_time_method="Median", interpolation='Linear')
    c_index = evaler.concordance(ties="All")[0]
    ibs_score = evaler.integrated_brier_score(num_points=10)
    _, dcal_hist = evaler.d_calibration()
    dcal = xcal_from_hist(dcal_hist)
    

    return c_index, ibs_score, dcal
    surv = surv_from_quantiles(predictors, target_alphas, surv_model.baseline_hazards_.index)

    durations = df_test['duration'].values
    events = df_test['event'].values
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    c_ind = ev.concordance_td('antolini')
    brier = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)

    return c_ind, brier, nbll


def csd(target_alphas, surv_model, df_train, df_cal, x_cal, x_test, durations_cal):
    args = {'mono_method': 'ceil', 'seed': 1, 'device': 'cuda:0', 'interpolate': 'Linear', 'decensor_method': 'sampling', 'n_quantiles': target_alphas.shape[0]}
    args = Namespace(**args)
    nc_model = QuantileRegressionNC(surv_model, args)
    icp = ConformalSurvDist(nc_model, condition=None, decensor_method=args.decensor_method,
                                    n_quantiles=args.n_quantiles)
    icp.quantile_levels = target_alphas
    
    icp.train_data = df_train.copy().rename(columns={'duration': 'time', 'event': 'event'})
    icp.feature_names = [col for col in icp.train_data.columns if col not in ['time', 'event']]
    # Add df_train to df_cal
    df_cal = pd.concat([df_cal, df_train])
    df_cal = df_cal.copy().rename(columns={'duration': 'time', 'event': 'event'})
    # Remove all columns that doent start with 'feat_' or are 'time' or 'event'
    df_cal.drop(columns=['T', 'C', 'mu_T', 'mu_C', 'std_T', 'std_C', 'index'], inplace=True, errors='ignore')
    icp.calibrate(df_cal)
    quan_levels, quan_preds = icp.predict(x_test)
    quan_preds = quan_preds[:, 1:]
    quan_levels = quan_levels[1:]
    return quan_preds.T

def base_model_cov(target_alphas, setting, surv_model, df_test, x_test):
    predictors = base_model(target_alphas, surv_model, x_test)

    if setting in range(1,7):
        T_test = df_test['T'].values
        coverage = (predictors <= T_test).mean(axis=1)
        return coverage, predictors.mean(axis=1)
    else:
        return predictors.mean(axis=1)

def base_model_metrics_estimation(target_alphas, setting, surv_model, df_test, x_test, df_train):
    predictors = base_model(target_alphas, surv_model, x_test)

    quant_preds = np.concatenate([np.zeros((predictors.T.shape[0], 1)), predictors.T], axis=1)
    quant_levels = np.concatenate([np.array([0]), target_alphas])

    evaler = QuantileRegEvaluator(quant_preds, quant_levels, df_test['duration'], df_test['event'], df_train['duration'], df_train['event'],
                                      predict_time_method="Median", interpolation='Linear')
    c_index = evaler.concordance(ties="All")[0]
    ibs_score = evaler.integrated_brier_score(num_points=10)
    hinge_abs = evaler.mae(method='Hinge', verbose=False, weighted=True)
    _, dcal_hist = evaler.d_calibration()
    dcal = xcal_from_hist(dcal_hist)
    
    return c_index, ibs_score, dcal

    surv = surv_from_quantiles(predictors, target_alphas, surv_model.baseline_hazards_.index)

    durations = df_test['duration'].values
    events = df_test['event'].values
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    c_ind = ev.concordance_td('antolini')
    brier = ev.integrated_brier_score(time_grid)
    nbll = ev.integrated_nbll(time_grid)

    return c_ind, brier, nbll

def base_model(target_alphas, surv_model, x_test):
    surv_test = surv_model.predict_surv_df(x_test)
    predictors = get_quantile_time(surv_test, target_alphas).values
    return predictors

def adaptive_conformal_cov(target_alphas, alphas, setting, failsafe, regimen, surv_model, surv_classifier, df_train, df_cal, df_test, x_train, x_cal, x_test, durations, events, early_event_model=None): #s=None):
    if regimen == 'fused':
        a_hats, include_proportion, predictors = adaptive_conformal(target_alphas, alphas, failsafe, regimen, surv_model, surv_classifier, x_cal, x_test, durations, events, early_event_model)
    else:
        a_hats, predictors = adaptive_conformal(target_alphas, alphas, failsafe, regimen, surv_model, surv_classifier, x_cal, x_test, durations, events, early_event_model)

    if setting in range(1,7):
        T_test = df_test['T'].values
        coverage = (predictors <= T_test).mean(axis=1)
        if regimen == 'fused':
            return coverage, a_hats, predictors.mean(axis=1), include_proportion
        else:
            return coverage, a_hats, predictors.mean(axis=1)
    else:
        if regimen == 'fused':
            return a_hats, predictors.mean(axis=1), include_proportion
        else:
            return a_hats, predictors.mean(axis=1)

def adaptive_conformal_metrics_estimation(target_alphas, alphas, setting, failsafe, regimen, surv_model
                                            , surv_classifier, df_train, df_cal, df_test, x_train, x_cal, x_test, durations, events, early_event_model=None):
    if regimen == 'fused':
        a_hats, include_proportion, predictors = adaptive_conformal(target_alphas, alphas, failsafe, regimen, surv_model
                                                                    , surv_classifier, x_cal, x_test, durations, events, early_event_model)
    else:
        a_hats, predictors = adaptive_conformal(target_alphas, alphas, failsafe, regimen, surv_model
                                                , surv_classifier, x_cal, x_test, durations, events, early_event_model)
        
    # surv = surv_from_quantiles(predictors, a_hats, surv_model.baseline_hazards_.index)

    # durations = df_test['duration'].values
    # events = df_test['event'].values
    # ev = EvalSurv(surv, durations, events, censor_surv='km')
    quant_preds = np.concatenate([np.zeros((predictors.T.shape[0], 1)), predictors.T], axis=1)
    quant_levels = np.concatenate([np.array([0]), target_alphas])

    evaler = QuantileRegEvaluator(quant_preds, quant_levels, df_test['duration'], df_test['event'], df_train['duration'], df_train['event'],
                                      predict_time_method="Median", interpolation='Linear')
    c_index = evaler.concordance(ties="All")[0]
    ibs_score = evaler.integrated_brier_score(num_points=10)
    _, dcal_hist = evaler.d_calibration()
    dcal = xcal_from_hist(dcal_hist)

    return c_index, ibs_score, dcal

def adaptive_conformal(target_alphas, alphas, failsafe, regimen, surv_model, surv_classifier, x_cal, x_test, durations, events, early_event_model):
    surv_df = surv_model.predict_surv_df(x_cal)
    q_alpha = get_quantile_time(surv_df, alphas)
        
    thresholds = q_alpha.apply(lambda col: (durations < col).astype(int), axis=1)

    event_probs_cal = surv_classifier.predict_proba(x_cal)[:, 1]
    weights_cal = 1. / event_probs_cal

    if regimen == 'fused':
        early_event_model_e_0 = early_event_model.predict_proba(np.concatenate([x_cal, np.zeros((x_cal.shape[0], 1))], axis=1))
        # For each array in the list, if it's a one coulmn array with value 1, convert it to a 2D array with 1 in the second column
        early_event_model_e_0 = [np.column_stack(((1 - arr) * np.ones_like(arr), arr * np.ones_like(arr))) if arr.shape[1] == 1 else arr for arr in early_event_model_e_0]
        early_event_model_e_0 = np.stack(early_event_model_e_0)[:, :, 1].T
        early_event_model_e_1 = early_event_model.predict_proba(np.concatenate([x_cal, np.ones((x_cal.shape[0], 1))], axis=1))
        early_event_model_e_1 = [np.column_stack(((1 - arr) * np.ones_like(arr), arr * np.ones_like(arr))) if arr.shape[1] == 1 else arr for arr in early_event_model_e_1]
        early_event_model_e_1 = np.stack(early_event_model_e_1)[:, :, 1].T

        include = early_event_model_e_0 < early_event_model_e_1

        weights_cal = np.where(include, 1, weights_cal[:, np.newaxis])
        weights_cal = np.where((events == 0)[:, np.newaxis] & (include == 0), 0, weights_cal)

    elif regimen == 'naive':
        weights_cal = np.ones_like(weights_cal)

    elif regimen == 'focus':
        weights_cal = np.where(events == 0, 0, weights_cal)

    alpha_hats = (thresholds*weights_cal.T).sum(axis=1) / weights_cal.sum(axis=0)
    if failsafe:
        weights_cal = np.ones_like(weights_cal)
        alpha_failsafe = (thresholds*weights_cal.T).sum(axis=1) / weights_cal.sum(axis=0)
        alpha_hats = np.minimum(alpha_hats.values, alpha_failsafe.values)

    alpha_diff = target_alphas - alpha_hats.to_numpy()[:, np.newaxis]
    smallest_pos = np.where(alpha_diff > 0, 1, -1. * np.inf).cumsum(axis=0).argmax(axis=0)
    a_hats = alphas[smallest_pos]
    if regimen == 'fused':
        include_proportion = include[smallest_pos].mean(axis=1)

    surv_test = surv_model.predict_surv_df(x_test)
    predictors = get_quantile_time(surv_test, a_hats).values
    if regimen == 'fused':
        return a_hats,include_proportion,predictors
    else:
        return a_hats,predictors