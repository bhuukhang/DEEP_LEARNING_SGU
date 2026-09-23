import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "house_price" / "data"
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
SUBMISSION_PATH = EXPERIMENT_DIR / "submission.csv"
for directory in (MODEL_DIR, RESULTS_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "SalePrice"
VALIDATION_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kaggle_rmsle(predictions, targets):
    predictions = np.maximum(predictions, 0)
    return float(np.sqrt(np.mean((np.log1p(predictions) - np.log1p(targets)) ** 2)))


def raw_rmse(predictions, targets):
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def decode_target(values, config):
    if config["target_log"]:
        values = values * config.get("target_std", 1.0) + config.get("target_mean", 0.0)
        return np.expm1(values)
    return values


def add_features(frame):
    frame = frame.copy()
    frame["TotalSF"] = frame["TotalBsmtSF"].fillna(0) + frame["1stFlrSF"].fillna(0) + frame["2ndFlrSF"].fillna(0)
    frame["TotalBathrooms"] = frame["FullBath"].fillna(0) + 0.5 * frame["HalfBath"].fillna(0)
    frame["TotalRoomsAndBedrooms"] = frame["TotRmsAbvGrd"].fillna(0) + frame["BedroomAbvGr"].fillna(0)
    frame["TotalPorchSF"] = frame[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].fillna(0).sum(axis=1)
    frame["HouseAgeAtSale"] = frame["YrSold"] - frame["YearBuilt"]
    frame["RemodAgeAtSale"] = frame["YrSold"] - frame["YearRemodAdd"]
    frame["GarageAgeAtSale"] = frame["YrSold"] - frame["GarageYrBlt"].fillna(frame["YearBuilt"])
    frame["OverallQual_TotalSF"] = frame["OverallQual"] * frame["TotalSF"]
    frame["OverallQual_GrLivArea"] = frame["OverallQual"] * frame["GrLivArea"]
    return frame


def prepare_data(train_df, test_df, improved):
    train_ids = train_df["Id"].copy()
    test_ids = test_df["Id"].copy()
    train_df = train_df.drop(columns=[TARGET, "Id", "Alley", "PoolQC", "Fence", "MiscFeature"], errors="ignore")
    test_df = test_df.drop(columns=["Id", "Alley", "PoolQC", "Fence", "MiscFeature"], errors="ignore")
    if improved:
        train_df = add_features(train_df)
        test_df = add_features(test_df)

    if not improved:
        train_categorical = ["FireplaceQu", "GarageType", "GarageFinish", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageQual", "GarageCond"]
        train_numeric = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]
        test_categorical = train_categorical + ["MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual", "Functional", "SaleType"]
        test_numeric = train_numeric + ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
        for column in train_categorical:
            train_df[column] = train_df[column].fillna(train_df[column].mode()[0])
        for column in train_numeric:
            train_df[column] = train_df[column].fillna(train_df[column].mean())
        for column in test_categorical:
            test_df[column] = test_df[column].fillna(test_df[column].mode()[0])
        for column in test_numeric:
            test_df[column] = test_df[column].fillna(test_df[column].mean())
        categorical_columns = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"]
    else:
        categorical_columns = None
    combined = pd.concat([train_df, test_df], ignore_index=True)
    categorical_columns = categorical_columns or combined.select_dtypes(include=["object"]).columns.tolist()
    for column in categorical_columns:
        combined[column] = combined[column].fillna("Missing").astype(str)
    combined = pd.get_dummies(combined, columns=categorical_columns, drop_first=False, dtype=np.float32)
    if not improved:
        combined = combined.loc[:, ~combined.columns.duplicated()]
    train_processed = combined.iloc[: len(train_df)].copy()
    test_processed = combined.iloc[len(train_df) :].copy()

    if improved:
        numeric_columns = train_processed.select_dtypes(exclude=["object"]).columns
        medians = train_processed[numeric_columns].median()
        train_processed[numeric_columns] = train_processed[numeric_columns].fillna(medians)
        test_processed[numeric_columns] = test_processed[numeric_columns].fillna(medians)
        scaler = StandardScaler()
        train_values = scaler.fit_transform(train_processed).astype(np.float32)
        test_values = scaler.transform(test_processed).astype(np.float32)
    else:
        train_processed = train_processed.fillna(train_processed.mean(numeric_only=True)).fillna(0)
        test_processed = test_processed.fillna(train_processed.mean(numeric_only=True)).fillna(0)
        train_values = train_processed.astype(np.float32).to_numpy()
        test_values = test_processed.astype(np.float32).to_numpy()
    return train_ids, test_ids, train_values, test_values


class HousePriceMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, batch_norm):
        super().__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def train_one(x_train, y_train, x_valid, y_valid, config):
    set_seed(SEED)
    model = HousePriceMLP(x_train.shape[1], config["hidden_sizes"], config["dropout"], config["batch_norm"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(1)), batch_size=config["batch_size"], shuffle=True)
    valid_features = torch.from_numpy(x_valid).to(DEVICE)
    valid_target = torch.from_numpy(y_valid).to(DEVICE)
    best_score = float("inf")
    best_state = None
    patience_counter = 0
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        for batch_features, batch_target in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features.to(DEVICE)), batch_target.to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(valid_features).squeeze(1).cpu().numpy()
        score_predictions = decode_target(predictions, config)
        score_targets = decode_target(y_valid, config)
        score = kaggle_rmsle(score_predictions, score_targets)
        history.append({"epoch": epoch + 1, "validation_rmsle": score})
        scheduler.step(score)
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(valid_features).squeeze(1).cpu().numpy()
    predictions = decode_target(predictions, config)
    score_targets = decode_target(y_valid, config)
    return model, history, best_score, raw_rmse(predictions, score_targets)


def main():
    set_seed()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_ids, test_ids, base_features, base_test = prepare_data(train_df, test_df, improved=False)
    _, _, improved_features, improved_test = prepare_data(train_df, test_df, improved=True)
    target = train_df[TARGET].to_numpy(dtype=np.float32)
    split_indices = train_test_split(np.arange(len(target)), test_size=VALIDATION_SIZE, random_state=SEED)
    train_indices, valid_indices = split_indices
    base_scaler = StandardScaler().fit(base_features[train_indices])
    base_scaled = base_scaler.transform(base_features).astype(np.float32)
    base_test_scaled = base_scaler.transform(base_test).astype(np.float32)

    experiments = [
        {"id": "EXP-01", "name": "v2 reproduction", "features": base_features, "test": base_test, "target_log": False, "hidden_sizes": [256, 128], "dropout": 0.0, "batch_norm": False, "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 32, "epochs": 100, "early_stopping_patience": 100},
        {"id": "EXP-02", "name": "scaled features and log target", "features": base_scaled, "test": base_test_scaled, "target_log": True, "hidden_sizes": [256, 128], "dropout": 0.0, "batch_norm": False, "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 32, "epochs": 250, "early_stopping_patience": 35},
        {"id": "EXP-03", "name": "engineered and scaled features", "features": improved_features, "test": improved_test, "target_log": True, "hidden_sizes": [256, 128], "dropout": 0.10, "batch_norm": False, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 32, "epochs": 140, "early_stopping_patience": 25},
        {"id": "EXP-04", "name": "tuned regularized MLP", "features": improved_features, "test": improved_test, "target_log": True, "hidden_sizes": [256, 128], "dropout": 0.05, "batch_norm": False, "learning_rate": 0.0007, "weight_decay": 1e-4, "batch_size": 32, "epochs": 180, "early_stopping_patience": 30},
    ]
    log_values = np.log1p(target)
    log_mean = float(log_values[train_indices].mean())
    log_std = float(log_values[train_indices].std())
    experiments.append({"id": "EXP-05", "name": "normalized log target", "features": improved_features, "test": improved_test, "target_log": True, "target_mean": log_mean, "target_std": log_std, "hidden_sizes": [256, 128], "dropout": 0.05, "batch_norm": False, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 32, "epochs": 180, "early_stopping_patience": 30})
    results = []
    best = None
    for experiment in experiments:
        started = time.perf_counter()
        features = experiment["features"]
        target_values = np.log1p(target) if experiment["target_log"] else target
        if experiment.get("target_std"):
            target_values = (target_values - experiment["target_mean"]) / experiment["target_std"]
        model, history, validation_rmsle, validation_rmse = train_one(
            features[train_indices], target_values[train_indices], features[valid_indices], target_values[valid_indices], experiment
        )
        result = {key: value for key, value in experiment.items() if key not in {"features", "test"}}
        result.update({"input_size": features.shape[1], "validation_rmsle": validation_rmsle, "validation_rmse": validation_rmse, "epochs_run": len(history), "training_seconds": time.perf_counter() - started})
        results.append(result)
        if best is None or validation_rmsle < best["result"]["validation_rmsle"]:
            best = {"experiment": experiment, "result": result, "model": model, "history": history}
        print(f"{experiment['id']} {experiment['name']}: RMSLE={validation_rmsle:.6f}, RMSE={validation_rmse:.2f}, epochs={len(history)}")

    results_df = pd.DataFrame(results).sort_values("validation_rmsle")
    results_df["improvement_vs_baseline"] = results_df["validation_rmsle"].iloc[0] - results_df["validation_rmsle"]
    results_df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
    best_experiment = best["experiment"]
    best_result = best["result"]

    final_features = best_experiment["features"]
    final_target = np.log1p(target) if best_experiment["target_log"] else target
    if best_experiment.get("target_std"):
        final_target = (final_target - best_experiment["target_mean"]) / best_experiment["target_std"]
    final_model, _, _, _ = train_one(final_features, final_target, final_features[:1], final_target[:1], best_experiment)
    checkpoint = {"model_state_dict": final_model.state_dict(), "input_size": final_features.shape[1], "model_config": {key: best_experiment[key] for key in ("hidden_sizes", "dropout", "batch_norm")}, "preprocessing": {"improved": best_experiment["features"] is improved_features, "feature_count": final_features.shape[1], "target_transform": "log1p" if best_experiment["target_log"] else "identity", "seed": SEED, "validation_size": VALIDATION_SIZE, "random_state": SEED}, "best_validation_rmsle": best_result["validation_rmsle"], "best_validation_rmse": best_result["validation_rmse"]}
    torch.save(checkpoint, MODEL_DIR / "best_model.pt")
    config = {key: value for key, value in best_experiment.items() if key not in {"features", "test"}}
    config.update({"data_dir": str(DATA_DIR), "device": str(DEVICE), "selected_validation_rmsle": best_result["validation_rmsle"], "selected_validation_rmse": best_result["validation_rmse"], "baseline_validation_rmsle": float(results_df.loc[results_df["id"] == "EXP-01", "validation_rmsle"].iloc[0])})
    with (CONFIG_DIR / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    with torch.no_grad():
        test_output = final_model(torch.from_numpy(best_experiment["test"]).to(DEVICE)).squeeze(1).cpu().numpy()
    test_output = decode_target(test_output, best_experiment)
    submission = pd.DataFrame({"Id": test_ids, TARGET: np.maximum(test_output, 0)})
    submission.to_csv(SUBMISSION_PATH, index=False)
    pd.DataFrame(best["history"]).to_csv(RESULTS_DIR / "best_validation_history.csv", index=False)
    print(f"Best experiment: {best_result['id']} ({best_result['name']})")
    print(f"Best validation RMSLE: {best_result['validation_rmsle']:.6f}")
    print(f"Saved checkpoint: {MODEL_DIR / 'best_model.pt'}")
    print(f"Saved submission: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()