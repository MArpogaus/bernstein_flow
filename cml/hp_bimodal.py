#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : hyperopt.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-05-10 17:59:17 (Marcel Arpogaus)
# changed : 2024-02-05 17:31:37 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################

import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.activations import get_thetas_constrain_fn
from bimodal import run
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

if __name__ == "__main__":
    experiment_name = "hp_bimodal"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--10sec", help="run faster", action="store_true", dest="_10sec"
    )
    parser.add_argument(
        "--no-mlflow",
        help="disable mlfow tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument("--seed", help="random seed", default=1, type=int)

    args = parser.parse_args()

    # Ensure Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print("TFP Version", tfp.__version__)
    print("TF  Version", tf.__version__)

    metrics_path = os.path.join("metrics", experiment_name)
    artifacts_path = os.path.join("artifacts", experiment_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    common_model_kwds = {
        "scale_data": hp.choice("scale_data", [True, False]),
        "shift_data": hp.choice("shift_data", [True, False]),
        "thetas_constrain_fn": get_thetas_constrain_fn(
            low=-4, high=4, smooth_bounds=False, allow_flexible_bounds=False
        ),
    }

    space = {
        # hp.choice("scale_data_to_domain", [True, False]),
        "scale_data_to_domain": False,
        "model_kwds": dict(
            clip_to_bernstein_domain=hp.choice(
                "clip_to_bernstein_domain", [True, False]
            ),
            scale_base_distribution=hp.choice("scale_base_distribution", [True, False]),
            **common_model_kwds,
        ),
        "fit_kwds": {
            "batch_size": hp.choice("batch_size", [16, 32, 128, 512]),
            "epochs": 1000,
            "verbose": 0,
        },
    }

    mlflow.autolog()
    exp = mlflow.set_experiment(experiment_name)
    if os.environ.get("MLFLOW_RUN_ID", False):
        mlflow.start_run()
    else:
        mlflow.start_run(run_name="hypeopt")

    def F(params):
        if args._10sec:
            params["fit_kwds"].update({"epochs": 1})
            params["data_points"] = 25

        mlflow.start_run(
            experiment_id=exp.experiment_id, nested=mlflow.active_run() is not None
        )
        mlflow.log_param("seed", args.seed)
        mlflow.log_params(
            dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
        )
        mlflow.log_params(params["fit_kwds"])
        mlflow.log_params(params["model_kwds"])
        model, bf, hist = run(args.seed, params, metrics_path, artifacts_path)
        mlflow.log_artifacts(artifacts_path)

        loss = min(hist.history["val_loss"])
        flow = bf(model(tf.linspace(0.0, 1.0, 10)[..., None]))
        status = STATUS_OK
        if np.isnan(loss).any() or np.isnan(flow.sample(100)).any():
            status = STATUS_FAIL
        mlflow.end_run("FINISHED" if status == STATUS_OK else "FAILED")
        return {"loss": loss, "status": status}

    trials = Trials()
    best = fmin(
        F,
        space,
        algo=tpe.suggest,
        max_evals=2 if args._10sec else 50,
        trials=trials,
        rstate=np.random.default_rng(args.seed),
    )
    mlflow.log_params(best)
    mlflow.log_metric("best_score", min(trials.losses()))
    mlflow.log_dict(trials.trials, "trials.yaml")

    tpe_results = np.array(
        [
            [x["result"]["loss"]]
            + [h[0] if len(h) else np.nan for h in x["misc"]["vals"].values()]
            for x in trials.trials
        ]
    )

    hps = [c for c in trials.trials[0]["misc"]["vals"].keys()]
    columns = ["loss"] + hps
    tpe_results_df = pd.DataFrame(tpe_results, columns=columns)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    tpe_results_df[["loss"]].plot(ax=ax[0])
    tpe_results_df[hps].plot(ax=ax[1])
    mlflow.log_figure(fig, "hyperopt.png")
    mlflow.end_run()
