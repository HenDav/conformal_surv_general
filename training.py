import copy
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from model import create_mlp, MySoftPlus
from sklearn.ensemble import RandomForestClassifier
from datasets import generate_synthetic, get_real_dataset
from MakeSurvivalCalibratedAgain.utils.util_survival import survival_to_quantile


get_target = lambda df: (df['duration'].values, df['event'].values)

def train_models_get_data(setting, args, frac_early_cens=0.15, threshold_early_cens=0.5, frac_early_surv=0.15, threshold_early_surv=0.5, max_depth_w=4, eot=10):
    args = copy.deepcopy(args)
    if setting in range(1,7):
        df_train, df_test, df_cal, df_val, x_train, x_val, x_cal, x_test, in_features = generate_synthetic(setting, args, frac_early_cens, threshold_early_cens, frac_early_surv, threshold_early_surv, eot=eot)
    else:
        df_train, df_test, df_cal, df_val, x_train, x_val, x_cal, x_test, in_features = get_real_dataset(setting, args)

    y_train = get_target(df_train)
    y_val = get_target(df_val)
    val = x_val, y_val

    net = torch.nn.Sequential(
        create_mlp(in_features, args['num_nodes'], 1, args['batch_norm'],
                                args['dropout']),
        # MySoftPlus(low=1e-9)
    )

    surv_model = CoxPH(net, optimizer=tt.optim.Adam)
    surv_model.optimizer.set_lr(args['lr'])
    
    torch.autograd.set_detect_anomaly(True)
    log = surv_model.fit(x_train, y_train, args['batch_size'], args['epochs'], copy.deepcopy(args['callbacks']), args['verbose'],
                    val_data=val, val_batch_size=args['batch_size'])
    surv_model.compute_baseline_hazards()

    surv_classifier = RandomForestClassifier(max_depth=max_depth_w)
    surv_classifier.fit(x_train, y_train[1])

    return surv_model, surv_classifier, df_train, df_cal, df_test, x_train, x_cal, x_test

def get_quantile_time(df, alphas):
    surv_prob = df.values.T
    time_coordinates = df.index.values
    if time_coordinates[0] != 0:
        time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
        surv_prob = np.concatenate([np.ones([surv_prob.shape[0], 1]), surv_prob], 1)
    time_coordinates = np.repeat(time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0)
    quantile_times = survival_to_quantile(surv_prob, time_coordinates, alphas, 'Linear')
    quantile_times = pd.DataFrame(quantile_times.T, columns=df.columns, index=np.round(1 - alphas, 3))
    return quantile_times

    # Ensure alphas is a NumPy array
    alphas = np.asarray(alphas)
    
    # Convert DataFrame values to NumPy array before performing multi-dimensional indexing
    df_values = df.to_numpy()

    # Subtract each alpha from each value in the NumPy array
    diff = np.abs(df_values[:, :, np.newaxis] - (1 - alphas))

    # Find the indices of the minimum values along the time axis
    min_indices = diff.argmin(axis=0).T

    # Convert indices to corresponding time points (use NumPy array for indexing)
    quantile_times = pd.DataFrame(df.index.to_numpy()[min_indices], columns=df.columns, index=np.round(1 - alphas, 3))
    
    return quantile_times

def surv_from_quantiles(quantile_times, alphas, time_index):
    # Convert alphas to a survival probability array (1 - alpha)
    survival_probs = np.asarray(alphas)
    
    # Create an empty DataFrame to store reconstructed survival probabilities
    subject_index = pd.Index(range(quantile_times.shape[1]))
    reconstructed_df = pd.DataFrame(index=time_index,
                                    columns=subject_index)
    
    # Loop over each column (e.g., treatment groups) in the quantile_times DataFrame
    for col in subject_index:
        # Extract quantile times for the current column
        quantile_time_series = quantile_times[:, col]

        # Build the survival function by interpolating across time for each alpha level
        survival_values = np.interp(
            x=time_index,                         # times to interpolate
            xp=quantile_time_series[::-1],              # known quantile times
            fp=survival_probs[::-1],                    # known survival probabilities
            left=1.0,                             # survival probability before the first quantile time is 1
            right=0.0                             # survival probability after the last quantile time is 0
        )
        
        # Assign the interpolated survival values to the reconstructed DataFrame
        reconstructed_df[col] = survival_values
    
    return reconstructed_df

def train_early_event_models(surv_model, alphas, x_train, df_train, setting):
    surv_train = surv_model.predict_surv_df(x_train)
    q_alphas = get_quantile_time(surv_train, alphas)
    # model = MLPClassifier(hidden_layer_sizes=(32,))
    model = (RandomForestClassifier(max_depth=2))

    labels = np.zeros((x_train.shape[0], q_alphas.index.values.shape[0]))
    for i, alpha in enumerate(q_alphas.index.values):
        # print(f'shapes: {df_train["duration"].values.shape}, {q_alphas.loc[alpha].shape}')
        labels[:, i] = df_train['duration'].values < q_alphas.loc[alpha].values
    x_train_early = np.concatenate([x_train, df_train['event'].values[:, np.newaxis]], axis=1)
    model.fit(x_train_early, labels)
    return model
