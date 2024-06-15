import os
from sklearn.metrics import classification_report, roc_auc_score
import torch
import optuna
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from cnn_models import AE_optuna, AE_flexible
from matplotlib import pyplot as plt


def check_trial_dir(trial_name):
    """
    Function that takes as arguments the trial name and checks if it is a valid directory during
    the training phase of the CNN.
    """
    current_directory = os.getcwd()
    path_to_trial = os.path.join(current_directory, "..", "..", "trials", trial_name)
    path_to_config = os.path.join(path_to_trial, "config.json")
    path_to_meta = os.path.join(path_to_trial, "assets", "metadata.csv")
    path_to_expr = os.path.join(path_to_trial, "assets", "expression.csv")
    path_to_order = os.path.join(path_to_trial, "assets", "gene_ctype_order.csv")
    path_to_images = os.path.join(path_to_trial, "results", "images", "data_images")
    print("Checking if the trial directory is valid")
    if not os.path.exists(path_to_trial):
        raise FileNotFoundError(
            f"The trial name {trial_name} provided does not have a dedicated directory."
        )
    if not os.path.exists(path_to_config):
        raise FileNotFoundError(
            f"The directory {trial_name} does not have a dedicated config file."
        )
    if not os.path.exists(path_to_meta):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a metadata.csv file."
        )
    if not os.path.exists(path_to_expr):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a expression.csv file."
        )
    if not os.path.exists(path_to_order):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a gene_ctype_order.csv file."
        )
    if not os.path.exists(path_to_images):
        raise FileNotFoundError(
            f"The directory {path_to_images} does not exist or is not located inside the directory {trial_name}, results, images."
        )
    else:
        return (
            current_directory,
            path_to_trial,
            path_to_config,
            path_to_meta,
            path_to_expr,
            path_to_order,
            path_to_images,
        )


def acc(y_hat, y):
    """
    Function that returns the accuracy of the prediciton
    as a division between the number of correctly predicted
    values between the total predictions for each batch.
    """
    probs = y_hat
    winners = probs.argmax(dim=1)
    corrects = winners == y
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners


def batch_train(x, y, net, cost_func, cost_func_reconstruct, optimizer, config):
    """
    Function that does one iteration of the training of a batch, takes
    as arguments the batches of predictors (images) and responses, the CNN
    and cost function for the predictions and image reconstruction.
    It also takes as argument the optimizer used and config file.
    The function calculates metrics for assessing the quality of
    predicitons.
    Returns total loss, accuracy, precision, recall, f1-score and auc
    if binary outcome.
    """
    net.train()
    y_hat, recon_image, linear = net(x)
    loss = cost_func(y_hat, y)
    cost_func.zero_grad()
    cost_func_reconstruct.zero_grad()
    accuracy, pred_classes = acc(y_hat, y)
    auc = 0
    if config["trainer_binary/multi-class"] != "multi-class":
        try:
            auc = roc_auc_score(
                y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach()
            )
        except ValueError:
            auc = 0
    report = classification_report(
        digits=6,
        y_true=y.cpu().detach().numpy(),
        y_pred=pred_classes.cpu().detach().numpy(),
        output_dict=True,
        zero_division=0,
    )
    total_loss = loss
    total_loss.backward()
    optimizer.step()
    return (
        total_loss.item(),
        accuracy.item(),
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
        auc,
    )


def batch_valid(x, y, net, cost_func, config):
    """
    Function that does one iteration of the validation for a batch, takes
    as arguments the batches of predictors (images) and responses, the CNN
    and cost function for the predictions. It also takes as argument the
    config file and calculates metrics for assessing the quality of
    predicitons.
    Returns total loss, accuracy, precision, recall, f1-score and auc
    if binary outcome.
    """
    with torch.no_grad():
        net.eval()
        y_hat, recon_image, linear = net(x)
        loss = cost_func(y_hat, y)

        accuracy, pred_classes = acc(y_hat, y)
        auc = 0
        if config["trainer_binary/multi-class"] != "multi-class":
            try:
                auc = roc_auc_score(
                    y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach()
                )
            except ValueError:
                auc = 0
        report = classification_report(
            digits=6,
            y_true=y.cpu().detach().numpy(),
            y_pred=pred_classes.cpu().detach().numpy(),
            output_dict=True,
            zero_division=0,
        )
        total_loss = loss
        return (
            total_loss.item(),
            accuracy.item(),
            report["macro avg"]["precision"],
            report["macro avg"]["recall"],
            report["macro avg"]["f1-score"],
            auc,
        )


def saveModel(ep, optimizer, loss, best_model, path_to_trial, sex):
    """
    Helper function that takes as arguments epoch number, optimizer, loss
    value, model and the path to trial directory.
    Saves the state dictionary of optimizer, model, epoch and loss value.
    """
    torch.save(
        {
            "epoch": ep,
            "model_state_dict": best_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(
            path_to_trial, "results", "training", sex, "best_model", "best_model.pb"
        ),
    )


def run_trial(config, dataset, sex, path_to_trial, number_of_bins):

    def objective(trial):
        """ """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = trial.suggest_float("lr", config["min_lr"], config["max_lr"])
        lr_decay = trial.suggest_float(
            "lr_decay", config["min_lr_decay"], config["max_lr_decay"]
        )
        lr_decrease_start = trial.suggest_int(
            "lr_decrease_start",
            config["min_lr_decrease_start"],
            config["max_lr_decrease_start"],
        )

        weight_decay = trial.suggest_float(
            "weight_decay", config["min_weight_decay"], config["max_weight_decay"]
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "Adagrad"]#, "RMSprop", "SGD"]
        )

        batch_size = trial.suggest_int(
            "batch_size", config["min_batch_size"], config["max_batch_size"]
        )
        train_pct = trial.suggest_float(
            "train_pct", config["min_train_pct"], config["max_train_pct"]
        )
        train_size = int(len(dataset) * train_pct)
        test_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        trainLoader = DataLoader(
            train_set, batch_size=batch_size, num_workers=0, shuffle=True
        )
        valLoader = DataLoader(
            val_set, batch_size=batch_size, num_workers=0, shuffle=True
        )

        patience = trial.suggest_int(
            "patience", config["min_patience"], config["max_patience"]
        )
        loss_jitter_binary = trial.suggest_categorical("loss_jitter_binary", ["1", "0"])
        if loss_jitter_binary == "1":
            loss_jitter = 1
        elif loss_jitter_binary == "0":
            loss_jitter = np.random.uniform(0.99, 1.01)

        best_loss = float("+Inf")
        trigger_times = 0
        last_loss = 0
        cost_func = torch.nn.CrossEntropyLoss()
        cost_func_reconstruct = torch.nn.MSELoss()

        net = AE_optuna(trial, config, number_of_bins)
        net.to(device)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(
                net.parameters(), lr_decay=lr_decay, lr=lr, weight_decay=weight_decay
            )
        #elif optimizer_name == "RMSprop":
        #    optimizer = torch.optim.RMSprop(
        #        net.parameters(), lr=lr, weight_decay=weight_decay
        #    )
        #elif optimizer_name == "SGD":
        #    optimizer = torch.optim.SGD(
        #        net.parameters(), lr=lr, weight_decay=weight_decay
        #    )

        lambda1 = lambda epoch: 0.99**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        avg_loss_train = []
        avg_loss_val = []
        avg_f1_val = []

        for ep in tqdm(range(n_epochs), desc="Current trial progress"):
            (
                batch_train_f1,
                batch_val_auc,
                batch_train_auc,
                batch_train_loss,
                batch_val_f1,
                batch_val_loss,
                jittered_val_loss,
            ) = ([], [], [], [], [], [], [])

            for x, y_dat, id in trainLoader:
                loss, acc_train, precision, recall, f1, train_auc = batch_train(
                    x.to(device),
                    y_dat.to(device),
                    net,
                    cost_func,
                    cost_func_reconstruct,
                    optimizer,
                    config,
                )
                batch_train_loss.append(loss)
                batch_train_f1.append(f1)
                batch_train_auc.append(train_auc)
            avg_loss_train.append(np.array(batch_train_loss).mean())

            for x, y_dat, id in valLoader:
                loss, acc_val, precision, recall, f1, val_auc = batch_valid(
                    x.to(device), y_dat.to(device), net, cost_func, config
                )
                loss = loss
                batch_val_loss.append(loss)
                batch_val_f1.append(f1)
                batch_val_auc.append(val_auc)
                jittered_val_loss.append(loss * loss_jitter)
            avg_loss_val.append(np.array(batch_val_loss).mean())
            avg_f1_val.append(np.array(batch_val_f1).mean())
            if ep % 25 == 0:
                print(
                    f"Epoch {ep}: \n\t Train loss: {np.mean(batch_train_loss)} Train F1: {np.mean(batch_train_f1)} \n\t Valid loss: {np.mean(batch_val_loss)} Valid F1: {np.mean(batch_val_f1)}, \n\t LR: {optimizer.param_groups[0]['lr']}"
                )

            trial.report(np.mean(batch_val_loss), ep)

            if ep >= lr_decrease_start:
                scheduler.step()

            if np.mean(jittered_val_loss) < best_loss:
                best_loss = np.mean(batch_val_loss)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if np.mean(jittered_val_loss) > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early Stopping! At epoch {ep}")
                    raise optuna.exceptions.TrialPruned()

            else:
                trigger_times = 0

            last_loss = np.mean(batch_val_loss)

        return np.mean(avg_f1_val)

    n_epochs = config["number_of_epochs"]
    n_trials = config["number_of_optimization_trials"]
    random_seed = 33
    torch.manual_seed(random_seed)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    trial = study.best_trial
    print(f"Best trial for {sex}:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print(f"\nStatistics of the hyperparameter training for {sex}: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    df_hyp = study.trials_dataframe().drop(
        ["datetime_start", "datetime_complete", "duration"], axis=1
    )
    df_hyp = df_hyp.loc[df_hyp["state"] == "COMPLETE"]
    df_hyp = df_hyp.drop("state", axis=1)
    df_hyp = df_hyp.sort_values("value", ascending=False)
    df_hyp.to_csv(
        os.path.join(
            path_to_trial,
            "results",
            "training",
            sex,
            "hyperparameter_training",
            f"optimized_hyperparameters_{sex}.csv",
        ),
        index=False,
    )

    param_importance = optuna.importance.get_param_importances(study, target=None)
    df_param_importance = pd.DataFrame(columns=["Parameter", "Importance"])
    for key, value in param_importance.items():
        df_param_importance = pd.concat(
            [
                df_param_importance,
                pd.DataFrame({"Parameter": [key], "Importance": [value]}),
            ],
            ignore_index=True,
        )
    df_param_importance.to_csv(
        os.path.join(
            path_to_trial,
            "results",
            "training",
            sex,
            "hyperparameter_training",
            f"hyperparameter_importance_{sex}.csv",
        ),
        index=False,
    )


def train_optimized(config, dataset, sex, path_to_trial, number_of_bins):
    best_hyperparameters = pd.read_csv(
        os.path.join(
            path_to_trial,
            "results",
            "training",
            sex,
            "hyperparameter_training",
            f"optimized_hyperparameters_{sex}.csv",
        )
    )

    batch_size = int(best_hyperparameters["params_batch_size"][0])
    loss_jitter_binary = int(best_hyperparameters["params_loss_jitter_binary"][0])
    lr = float(best_hyperparameters["params_lr"][0])
    lr_decay = float(best_hyperparameters["params_lr_decay"][0])
    lr_decrease_start = int(best_hyperparameters["params_lr_decrease_start"][0])
    optimizer_name = best_hyperparameters["params_optimizer"][0]
    p_drop_ext = float(best_hyperparameters["params_p_drop_ext"][0])
    p_drop_pred = float(best_hyperparameters["params_p_drop_pred"][0])
    patience = float(best_hyperparameters["params_patience"][0])
    train_pct = float(best_hyperparameters["params_train_pct"][0])
    weight_decay = float(best_hyperparameters["params_weight_decay"][0])
    n_epochs = config["number_of_epochs"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_size = int(len(dataset) * train_pct)
    test_size = len(dataset) - train_size
    scale1 = int(best_hyperparameters["params_scale1"][0])
    scale2 = int(best_hyperparameters["params_scale2"][0])
    scale3 = int(best_hyperparameters["params_scale3"][0])
    scale4 = int(best_hyperparameters["params_scale4"][0])
    ext_scale = int(best_hyperparameters["params_ext_scale"][0])

    print("Train size: ", train_size)
    print("Test size: ", test_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainLoader = DataLoader(
        train_set, batch_size=batch_size, num_workers=0, shuffle=True
    )
    valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=True)

    best_loss = float("+Inf")
    best_model = None
    trigger_times = 0
    last_loss = 0
    net = AE_flexible(p_drop_ext, p_drop_pred, scale1, scale2, scale3, scale4,ext_scale, number_of_bins)
    cost_func = torch.nn.CrossEntropyLoss()
    cost_func_reconstruct = torch.nn.MSELoss()
    net.to(device)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(
            net.parameters(), lr_decay=lr_decay, lr=lr, weight_decay=weight_decay
        )
    #elif optimizer_name == "RMSprop":
    #    optimizer = torch.optim.RMSprop(
    #        net.parameters(), lr=lr, weight_decay=weight_decay
    #    )
    #elif optimizer_name == "SGD":
    #    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    lambda1 = lambda epoch: 0.99**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    avg_loss_train = []
    avg_loss_val = []

    for ep in tqdm(range(n_epochs), desc="Training progress"):
        (
            batch_train_f1,
            batch_val_auc,
            batch_train_auc,
            batch_train_loss,
            batch_val_f1,
            batch_val_loss,
            jittered_val_loss,
        ) = ([], [], [], [], [], [], [])

        for x, y_dat, id in trainLoader:
            loss, acc_train, precision, recall, f1, train_auc = batch_train(
                x.to(device),
                y_dat.to(device),
                net,
                cost_func,
                cost_func_reconstruct,
                optimizer,
                config,
            )
            batch_train_loss.append(loss)
            batch_train_f1.append(f1)
            batch_train_auc.append(train_auc)
        avg_loss_train.append(np.array(batch_train_loss).mean())

        for x, y_dat, id in valLoader:
            loss, acc_val, precision, recall, f1, val_auc = batch_valid(
                x.to(device), y_dat.to(device), net, cost_func, config
            )
            loss = loss
            batch_val_loss.append(loss)
            batch_val_f1.append(f1)
            batch_val_auc.append(val_auc)
            if loss_jitter_binary == 1:
                jittered_val_loss.append(loss)
            elif loss_jitter_binary == 0:
                jittered_val_loss.append(loss * np.random.uniform(0.99, 1.01))
        avg_loss_val.append(np.array(batch_val_loss).mean())

        print(
            f"Epoch {ep}: \n\t Train loss: {np.mean(batch_train_loss)} Train F1: {np.mean(batch_train_f1)} \n\t Valid loss: {np.mean(batch_val_loss)} Valid F1: {np.mean(batch_val_f1)}, \n\t LR: {optimizer.param_groups[0]['lr']}"
        )

        if ep >= lr_decrease_start:
            scheduler.step()

        if np.mean(jittered_val_loss) < best_loss:
            best_loss = np.mean(batch_val_loss)
            best_model = net
            print("Best loss!")

        if np.mean(jittered_val_loss) > last_loss:
            saveModel(
                ep, optimizer, np.mean(batch_val_loss), best_model, path_to_trial, sex
            )
            trigger_times += 1
            print("Trigger Times:", trigger_times)
            if trigger_times >= patience:
                print("Early Stopping!")
                break
        else:
            print("trigger times: 0")
            trigger_times = 0
        last_loss = np.mean(batch_val_loss)
    plt.clf()
    plt.plot(avg_loss_train, label="train_loss")
    plt.plot(avg_loss_val, label="val_loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            path_to_trial,
            "results",
            "training",
            sex,
            "loss_plot",
            f"loss_plot_{sex}.png",
        ),
        dpi=500,
    )


def train_fixed(config, dataset, sex, path_to_trial, number_of_bins):
    batch_size = config["batch_size"]
    loss_jitter_binary = config["loss_jitter_yes_no"]
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    lr_decrease_start = config["lr_decrease_start"]
    optimizer_name = config["optimizer"]
    p_drop_ext = config["p_drop_ext"]
    p_drop_pred = config["p_drop_pred"]
    patience = config["patience"]
    train_pct = config["train_pct"]
    weight_decay = config["params_weight_decay"]
    n_epochs = config["number_of_epochs"]
    scale1 = config["fixed_scale1"]
    scale2 = config["fixed_scale2"]
    scale3 = config["fixed_scale3"]
    scale4 = config["fixed_scale4"]
    ext_scale = config["fixed_ext_scale"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_size = int(len(dataset) * train_pct)
    test_size = len(dataset) - train_size
    print("Train size: ", train_size)
    print("Test size: ", test_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainLoader = DataLoader(
        train_set, batch_size=batch_size, num_workers=0, shuffle=True
    )
    valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=True)

    best_loss = float("+Inf")
    best_model = None
    trigger_times = 0
    last_loss = 0
    net = AE_flexible(p_drop_ext, p_drop_pred, scale1, scale2, scale3, scale4, ext_scale, number_of_bins)
    cost_func = torch.nn.CrossEntropyLoss()
    cost_func_reconstruct = torch.nn.MSELoss()
    net.to(device)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(
            net.parameters(), lr_decay=lr_decay, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    lambda1 = lambda epoch: 0.99**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    avg_loss_train = []
    avg_loss_val = []

    for ep in tqdm(range(n_epochs), desc="Training progress"):
        (
            batch_train_f1,
            batch_val_auc,
            batch_train_auc,
            batch_train_loss,
            batch_val_f1,
            batch_val_loss,
            jittered_val_loss,
        ) = ([], [], [], [], [], [], [])

        for x, y_dat, id in trainLoader:
            loss, acc_train, precision, recall, f1, train_auc = batch_train(
                x.to(device),
                y_dat.to(device),
                net,
                cost_func,
                cost_func_reconstruct,
                optimizer,
                config,
            )
            batch_train_loss.append(loss)
            batch_train_f1.append(f1)
            batch_train_auc.append(train_auc)
        avg_loss_train.append(np.array(batch_train_loss).mean())

        for x, y_dat, id in valLoader:
            loss, acc_val, precision, recall, f1, val_auc = batch_valid(
                x.to(device), y_dat.to(device), net, cost_func, config
            )
            loss = loss
            batch_val_loss.append(loss)
            batch_val_f1.append(f1)
            batch_val_auc.append(val_auc)
            if loss_jitter_binary == "no":
                jittered_val_loss.append(loss)
            elif loss_jitter_binary == "yes":
                jittered_val_loss.append(loss * np.random.uniform(0.99, 1.01))
        avg_loss_val.append(np.array(batch_val_loss).mean())

        print(
            f"Epoch {ep}: \n\t Train loss: {np.mean(batch_train_loss)} Train F1: {np.mean(batch_train_f1)} \n\t Valid loss: {np.mean(batch_val_loss)} Valid F1: {np.mean(batch_val_f1)}, \n\t LR: {optimizer.param_groups[0]['lr']}"
        )

        if ep >= lr_decrease_start:
            scheduler.step()

        if np.mean(jittered_val_loss) < best_loss:
            best_loss = np.mean(batch_val_loss)
            best_model = net
            print("Best loss!")

        if np.mean(jittered_val_loss) > last_loss:
            saveModel(
                ep, optimizer, np.mean(batch_val_loss), best_model, path_to_trial, sex
            )
            trigger_times += 1
            print("Trigger Times:", trigger_times)
            if trigger_times >= patience:
                print("Early Stopping!")
                break
        else:
            print("trigger times: 0")
            trigger_times = 0
        last_loss = np.mean(batch_val_loss)
    plt.clf()
    plt.plot(avg_loss_train, label="train_loss")
    plt.plot(avg_loss_val, label="val_loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            path_to_trial,
            "results",
            "training",
            sex,
            "loss_plot",
            f"loss_plot_{sex}.png",
        ),
        dpi=500,
    )
