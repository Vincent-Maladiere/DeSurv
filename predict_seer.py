# %%

from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)

path_seer = (
    "/Users/vincentmaladiere/dev/hazardous/hazardous/data/"
    "seer_cancer_cardio_raw_data.txt"
)

X, y = load_seer(
    input_path=path_seer,
    survtrace_preprocessing=True,
    return_X_y=True,
)
X = X.dropna()
y = y.iloc[X.index]
X.shape, y.shape
# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y["event"],
    random_state=0,
)
X_train_, X_val, y_train_, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.5,
    stratify=y_train["event"],
    random_state=0,
)

duration_train = y_train_["duration"]
event_train = y_train_["event"]

duration_val = y_val["duration"]
event_val = y_val["event"]

duration_test = y_test["duration"]
event_test = y_test["event"]

print(X_train_.shape, X_val.shape)

# %%
from hazardous.survtrace._encoder import SurvFeatureEncoder

enc = SurvFeatureEncoder(
    categorical_columns=CATEGORICAL_COLUMN_NAMES,
    numeric_columns=NUMERIC_COLUMN_NAMES,
)
X_train_ = enc.fit_transform(X_train_)
X_val = enc.transform(X_val)
X_test = enc.transform(X_test)
X_train_.shape, X_val.shape

# %%
t_max = y["duration"].max()
duration_train /= t_max
duration_val /= t_max
duration_test /= t_max

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader


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
    
dataloader_train = get_dataloader(X_train_, duration_train, event_train)
dataloader_val = get_dataloader(X_val, duration_val, event_val)
dataloader_test = get_dataloader(X_test, duration_test, event_test)


# %%
from src.classes import ODESurvMultiple

xdim = X_train_.shape[1]
lr = 1e-3
hidden_dim = 32
model = ODESurvMultiple(lr, xdim, hidden_dim, num_risks=3)
model

# %%

from time import time

train = False
if train:
    tic = time()
    model.optimize(
        dataloader_train,
        n_epochs=300,
        logging_freq=1,
        data_loader_val=dataloader_val,
        max_wait=20,
    )
    torch.save(model.state_dict(), "tst_model")
    toc = time()
    print(f"{toc - tic:.2f}s")
else:
    state_dict = torch.load("tst_model")
    model.load_state_dict(state_dict)
    model.eval()

# %%
import numpy as np


def get_y_pred(model, X_test, horizon):
    model.eval()
    time_grid = np.linspace(
        y["duration"].min(),
        y["duration"].max(),
        100,
    )
    time_grid = np.quantile(time_grid, horizon)
    time_grid /= t_max
    
    n_test_samples = X_test.shape[0]
    n_time_grid = time_grid.size

    with torch.no_grad():
        t_ = torch.tensor(
            np.concatenate(
                [time_grid] * n_test_samples, axis=0,
            ),
            dtype=torch.float32,
        )
        x_ = torch.tensor(
            np.repeat(
                X_test,
                [n_time_grid] * n_test_samples, axis=0,
            ),
            dtype=torch.float32,
        )

        y_pred, pi = model.predict(x_, t_)
        y_pred = y_pred.reshape(n_test_samples, -1, 3).permute(2, 0, 1)
        y_pred = y_pred.detach().numpy()
        surv_pred = 1 - y_pred.sum(axis=0)[None, :, :]
        y_pred = np.concatenate([surv_pred, y_pred], axis=0)

        return time_grid * t_max, y_pred

# %%
horizon = [.25, .50, .75]
time_grid, y_pred = get_y_pred(model, X_test, horizon)
print(y_pred.shape)
print(y_pred)
print(time_grid)

# %%
from collections import defaultdict
from sksurv.metrics import concordance_index_ipcw


def get_c_index(y_train, y_test, y_pred, time_grid, n_events=3):

    print(y_train["duration"].max())
    print(y_test["duration"].max())
    print(time_grid.max())

    c_indexes = defaultdict(list)

    y_train_binary = y_train.copy()
    y_test_binary = y_test.copy()

    for event_idx in range(n_events):

        y_train_binary["event"] = (y_train["event"] == (event_idx + 1)) 
        y_test_binary["event"] = (y_test["event"] == (event_idx + 1))

        et_train = make_recarray(y_train_binary)
        et_test = make_recarray(y_test_binary)

        for time_idx in range(len(time_grid)):
            y_pred_at_t = y_pred[event_idx+1][:, time_idx]
            tau = time_grid[time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(                
                et_train,
                et_test,
                y_pred_at_t,
                tau=tau,
            )
            c_indexes[event_idx+1].append(round(ct_index, 3))
    
    return c_indexes


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


get_c_index(y_train, y_test, y_pred, time_grid)

# %%

horizons = np.linspace(0., 1., 20)
time_grid, y_pred = get_y_pred(model, X_test, horizons)
print(y_pred.shape)
print(time_grid)
y_pred

# %%
from hazardous.metrics._brier_score import integrated_brier_score_incidence

n_events = 3
all_ibs = []

for event_idx in range(n_events):
    ibs = integrated_brier_score_incidence(
        y_train,
        y_test,
        y_pred[event_idx+1],
        time_grid,
        event_of_interest=event_idx + 1,
    )
    all_ibs.append(ibs)

print(all_ibs)


# %%
horizon = [.125, .25, .375, .5, .625, .75]    
time_grid, y_pred = get_y_pred(model, X_test, horizon)
accuracy_in_time = []

for time_idx in range(len(time_grid)):

    y_pred_time = y_pred[:, :, time_idx]
    mask = (y_test["event"] == 0) & (y_test["duration"] < time_grid[time_idx])
    y_pred_time = y_pred_time[:, ~mask]
    
    y_pred_class = y_pred_time.argmax(axis=0)
    y_test_class = y_test["event"] * (y_test["duration"] < time_grid[time_idx])
    y_test_class = y_test_class.loc[~mask]

    score = (y_test_class.values == y_pred_class).mean()
    accuracy_in_time.append(score)

accuracy_in_time
# %%
