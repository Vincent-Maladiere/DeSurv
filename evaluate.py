# %%
from pathlib import Path
from time import time
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_ipcw

from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.utils import make_time_grid
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

from src.classes import ODESurvMultiple

PATH_SCORES = Path("../benchmark/scores")
PATH_SEER = Path("../hazardous/data/seer_cancer_cardio_raw_data.txt")
WEIBULL_PARAMS = {
    "n_events": 3,
    "n_samples": 20_000,
    "censoring_relative_scale": 1.5,
    "complex_features": False,
    "independent_censoring": False,
}
SEEDS = range(5)
N_STEPS_TIME_GRID = 20
MODEL_NAME = "DeSurv"


def run_evaluation(dataset_name):

    all_scores = []

    for random_state in tqdm(SEEDS):
        bunch, dataset_params = get_dataset(dataset_name, random_state)
        model, fit_time = get_model(bunch, train=True)
        
        scores = evaluate(
            model,
            bunch,
            dataset_name,
            dataset_params=dataset_params,
            model_name=MODEL_NAME,
        )
        scores["fit_time"] = fit_time
        
        all_scores.append(scores)
        
        path_dir = PATH_SCORES / "raw" / MODEL_NAME
        path_dir.mkdir(parents=True, exist_ok=True)
        path_raw_scores = path_dir / f"{dataset_name}.json"
        json.dump(all_scores, open(path_raw_scores, "w"))
    

def evaluate(
    model, bunch, dataset_name, dataset_params, model_name, verbose=True
):
    """Evaluate a model against its test set.
    """
    X_train, y_train = bunch["X_train"], bunch["y_train"]
    X_test, y_test = bunch["X_test"], bunch["y_test"]

    n_events = np.unique(y_train["event"]).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": X_train.shape[0],
        "n_cols": X_train.shape[1],
        "censoring_rate": (y_train["event"] == 0).mean(),
        **dataset_params,
    }

    time_grid = make_time_grid(y_test["duration"], n_steps=N_STEPS_TIME_GRID)
    y_pred, predict_time = get_y_pred(model, time_grid, bunch)

    print(f"{time_grid=}")

    scores["time_grid"] = time_grid.round(4).tolist()
    scores["y_pred"] = y_pred.round(4).tolist()
    scores["predict_time"] = round(predict_time, 2)

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []

    if verbose:
        print("Computing Brier scores, IBS and C-index")

    for event_id in range(1, n_events+1):

        # Brier score and IBS
        if dataset_name == "weibull":
            # Use oracle metrics with the synthetic dataset.
            ibs = integrated_brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring,
                scale_censoring=bunch.scale_censoring,
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring,
                scale_censoring=bunch.scale_censoring,
                event_of_interest=event_id,  
            )
        else:
            ibs = integrated_brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )   
            
        print(f"event{event_id} ibs: {ibs}")
            
        event_specific_ibs.append({
            "event": event_id,
            "ibs": round(ibs, 4),
        })
        event_specific_brier_scores.append({
            "event": event_id,
            "time": list(time_grid.round(2)),
            "brier_score": list(brier_scores.round(4)),
        })

        # C-index
        y_train_binary = y_train.copy()
        y_test_binary = y_test.copy()

        y_train_binary["event"] = (y_train["event"] == event_id)
        y_test_binary["event"] = (y_test["event"] == event_id)

        truncation_quantiles = [0.25, 0.5, 0.75]
        taus = np.quantile(time_grid, truncation_quantiles)
        taus = tqdm(
            taus,
            desc=f"c-index at tau for event {event_id}",
            total=len(taus),
        )
        c_indices = []
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred[event_id][:, tau_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                make_recarray(y_train_binary),
                make_recarray(y_test_binary),
                y_pred_at_t,
                tau=tau,
            )
            c_indices.append(round(ct_index, 4))
        print(f"event{event_id} c_indices: {c_indices}")

        event_specific_c_index.append({
            "event": event_id,
            "time_quantile": truncation_quantiles,
            "c_index": c_indices,
        })

    scores.update({
        "event_specific_ibs": event_specific_ibs,
        "event_specific_brier_scores": event_specific_brier_scores,
        "event_specific_c_index": event_specific_c_index,
    })

    if is_competing_risk:
        # Accuracy in time
        if verbose:
            print("Computing accuracy in time")

        truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        times = np.quantile(time_grid, truncation_quantiles)
        accuracy = []
        
         # TODO: put it into a function in hazardous._metrics
        for time_idx in range(len(times)):
            y_pred_at_t = y_pred[:, :, time_idx]
            mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
            y_pred_class = y_pred_at_t[:, ~mask].argmax(axis=0)
            y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
            y_test_class = y_test_class.loc[~mask]
            accuracy.append(
                round(
                    (y_test_class.values == y_pred_class).mean(),
                    4
                )
            )
        scores["accuracy_in_time"] = {
            "time_quantile": truncation_quantiles,
            "accuracy": accuracy,
        }

    else:
        # Yana loss
        if verbose:
            print("Computing Censlog")

        censlog = CensoredNegativeLogLikelihoodSimple().loss(
            y_pred, y_test["duration_test"], y_test["event"], time_grid
        )
        scores["censlog"] = round(censlog, 4)        

    return scores


def get_dataset(dataset_name, random_state):
    bunch, dataset_params = load_dataset(dataset_name, random_state)
    X, y = bunch["X"], bunch["y"]
    print(f"{X.shape=}, {y.shape=}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=random_state,
    )
    X_train_, X_val, y_train_, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.5,
        stratify=y_train["event"],
        random_state=random_state,
    )

    duration_train = y_train_["duration"].copy()
    event_train = y_train_["event"]

    duration_val = y_val["duration"].copy()
    event_val = y_val["event"]

    duration_test = y_test["duration"].copy()
    event_test = y_test["event"]

    print(f"{X_train_.shape=}, {X_val.shape=}")

    enc = SurvFeatureEncoder(
        categorical_columns=CATEGORICAL_COLUMN_NAMES,
        numeric_columns=NUMERIC_COLUMN_NAMES,
    )
    X_train_ = enc.fit_transform(X_train_)
    X_val = enc.transform(X_val)
    X_test = enc.transform(X_test)
    print(f"{X_train_.shape=}, {X_val.shape=}")
    
    t_max = y["duration"].max()
    duration_train /= t_max
    duration_val /= t_max
    duration_test /= t_max

    dataloader_train = get_dataloader(X_train_, duration_train, event_train)
    dataloader_val = get_dataloader(X_val, duration_val, event_val)

    n_features = X_train_.shape[1]
    
    n_events = len(set(np.unique(event_train)) - {0})

    bunch.update({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "n_features": n_features,
        "t_max": t_max,
        "n_events": n_events
    })
    
    return bunch, dataset_params
    

def load_dataset(dataset_name, random_state):

    dataset_params = {"random_state": random_state}

    if dataset_name == "seer":
        X, y = load_seer(
            input_path=PATH_SEER,
            survtrace_preprocessing=True,
            return_X_y=True,
        )
        X = X.dropna()
        y = y.iloc[X.index]
        bunch = {"X": X, "y": y}

    elif dataset_name == "weibull":
        dataset_params.update(WEIBULL_PARAMS)
        bunch = make_synthetic_competing_weibull(**dataset_params)
    else:
        raise ValueError(dataset_name)

    return bunch, dataset_params


def get_dataloader(X, duration, event, batch_size=32):
    dataset = TensorDataset(
        torch.tensor(X.to_numpy(), dtype=torch.float32),
        torch.tensor(duration.to_numpy(), dtype=torch.float32),
        torch.tensor(event.to_numpy(), dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    ) 


def get_model(bunch, train=False):
    
    model = init_model(bunch["n_features"])
    
    time_to_fit = None
    if train:
        tic = time()
        model.optimize(
            bunch["dataloader_train"],
            n_epochs=200,
            logging_freq=1,
            data_loader_val=bunch["dataloader_val"],
            max_wait=20,
        )
        torch.save(model.state_dict(), "tst_model")
        time_to_fit = time() - tic
        print(f"{time_to_fit=:.2f}s")

    else:
        state_dict = torch.load("tst_model")
        model.load_state_dict(state_dict)
        model.eval()

    return model, time_to_fit


def init_model(n_features, lr=1e-3, hidden_dim=32):
    return ODESurvMultiple(lr, n_features, hidden_dim, num_risks=3)


def get_y_pred(model, time_grid, bunch):
    model.eval()

    time_grid = time_grid.copy()
    time_grid /= bunch["t_max"]
    
    n_test_samples = bunch["X_test"].shape[0]
    n_time_grid = time_grid.size

    tic = time()

    with torch.no_grad():
        t_ = torch.tensor(
            np.concatenate(
                [time_grid] * n_test_samples, axis=0,
            ),
            dtype=torch.float32,
        )
        x_ = torch.tensor(
            np.repeat(
                bunch["X_test"],
                [n_time_grid] * n_test_samples, axis=0,
            ),
            dtype=torch.float32,
        )

        y_pred, pi = model.predict(x_, t_)
        y_pred = y_pred.reshape(n_test_samples, -1, 3).permute(2, 0, 1)
        y_pred = y_pred.detach().numpy()
        surv_pred = 1 - y_pred.sum(axis=0)[None, :, :]
        y_pred = np.concatenate([surv_pred, y_pred], axis=0)

    predict_time = time() - tic

    return y_pred, predict_time


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


# %%

if __name__ == "__main__":
    run_evaluation("seer")
# %%
