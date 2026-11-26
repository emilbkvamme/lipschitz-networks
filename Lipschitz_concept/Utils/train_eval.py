import copy
import hashlib
import math
import multiprocessing as mp
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

##### UTILITY FUNCTIONS #####
# The following functions are just utility functions to avoid code repetition. Their format is based on personal preference

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train() 
    train_loss = 0.0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad() # Must be set to zero each iteration to avoid exploding gradients
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward() # Backpropagation to compute gradients
        optimizer.step() # Update the model parameters

    return

def evaluate(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode, this makes the model behave faster as no dropout, batchnorm, or gradient calculations are needed
    val_loss = 0.0
    total_samples = 0

    # The weights are fixed here, so the normalized (parametrized) weights are computed once instead of for every batch
    with torch.no_grad(), parametrize.cached(): # No gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            val_loss += loss_fn(y_pred, y).item()*X.size(0) # Multiply by batch size to get the total loss for this batch
            total_samples += X.size(0)

    return val_loss / total_samples

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader=None, verbose=True,
          scheduler=None, eval_every=1, history=None):
    # train_eval_loader: optional loader over the training data used only for computing the training loss.
    # Passing one with a large batch size (or over a fixed subsample of the training events) makes this much cheaper than
    # iterating the (small batch) train_loader again. The training loss is only recorded: nothing below depends on it.
    # scheduler: optional ReduceLROnPlateau, which lowers the learning rate when the validation loss stops improving.
    # eval_every: evaluate (losses, scheduler, early stopping) only every this many epochs, and after the last one.
    # max_iter and the scheduler's patience then count evaluations, not epochs. Useful when an epoch is only a few steps.
    # history: optional dictionary, filled with "eval_epochs" (the epoch, from 1, of every evaluation) and
    # "best_epoch" (the epoch whose weights the model ends with).
    # At the end the model always holds the weights of the evaluation with the best validation loss, whether early
    # stopping ended the training or the num_epochs cap did.
    if train_eval_loader is None:
        train_eval_loader = train_loader
    eval_every = max(1, int(eval_every))

    train_losses = []
    val_losses = []
    eval_epochs = []

    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_state = None # Weights of the model at the evaluation with the best validation loss
    best_epoch = None

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        if (epoch + 1) % eval_every != 0 and epoch != num_epochs - 1:
            continue
        train_loss = evaluate(model, train_eval_loader, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_epochs.append(epoch + 1)

        if verbose and (len(val_losses) % 50 == 1 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= max_iter:
            if verbose:
                print(f"Early stopping triggered. Restoring the weights from epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state) # Go back to the best weights instead of keeping the latest ones
    if history is not None:
        history["eval_epochs"] = eval_epochs
        history["best_epoch"] = best_epoch

    return train_losses, val_losses

def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad(), parametrize.cached():
        for X, _ in dataloader:
            X = X.to(device)
            y_pred = model(X)
            predictions.append(y_pred.cpu().numpy())
    return np.concatenate(predictions, axis=0)


##### FAST BATCHES #####
# A DataLoader over a TensorDataset picks every event on its own and stacks them again into a batch, which for these
# small networks takes longer than the training step itself. TensorBatches cuts the batches straight out of the tensors.
# It draws the same random numbers from the generator as DataLoader(shuffle=True, generator=generator) does, so the
# batches, and therefore the trained networks, are exactly the same as with the DataLoader.

class TensorBatches:
    def __init__(self, X, y, batch_size, shuffle=False, generator=None):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.X)
        if self.shuffle:
            # DataLoader first draws a base seed for its workers, then RandomSampler draws one permutation for the epoch,
            # and a second (unused) one when the epoch is exhausted. The same draws keep the generator in step with it.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            order = torch.randperm(n, generator=self.generator)
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                yield self.X[batch], self.y[batch]
            torch.randperm(n, generator=self.generator)
        else:
            for start in range(0, n, self.batch_size):
                yield self.X[start:start + self.batch_size], self.y[start:start + self.batch_size]


##### CORES #####
# How many cores a run may use. max_cores caps the number of networks trained at the same time (one process each),
# so a large scan can run without taking the whole machine.

def available_cores():
    """The cores this process may use. On a cluster that is only the cores the job asked for, while os.cpu_count()
    gives all cores of the machine. macOS has no sched_getaffinity, but there all cores can be used anyway."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 2

def n_parallel_workers(n_jobs, max_cores=None, n_workers=None, default=None):
    """
    The number of worker processes for n_jobs independent trainings:
        at most n_jobs, at most max_cores (None: no cap) and at most n_workers (None: no cap).
    With neither max_cores nor n_workers given, default(n_cores) decides (n_cores = available_cores()).
    """
    if max_cores is None and n_workers is None:
        if default is None:
            default = lambda n_cores: n_cores
        return max(1, min(n_jobs, default(available_cores())))
    limit = min(x for x in (max_cores, n_workers) if x is not None)
    return max(1, min(n_jobs, limit))

def limit_cores(max_cores):
    """
    Caps the threads PyTorch uses in THIS process (e.g. a notebook or script training one large network, or computing
    slopes on all events) to max_cores. None leaves the default (all cores). The parallel sweeps below set the
    threads of their worker processes themselves.
    """
    if max_cores is not None:
        torch.set_num_threads(max(1, int(max_cores)))

def int_from_argv(flag, default=None, argv=None):
    """Reads "<flag> N" (e.g. "--batch-size 4096") from the command line of a script (default if it is not given)."""
    argv = sys.argv if argv is None else argv
    if flag in argv:
        return int(argv[argv.index(flag) + 1])
    return default

def scaled_lr(base_lr, batch_size, base_batch_size=1024):
    """
    The learning rate for batch_size, from one that works for base_batch_size: scaled with the square root of the batch
    size ratio (Adam: a 4x larger batch averages the gradient over 4x more events, so its noise is 2x smaller and a 2x
    larger step is about as stable). base_lr itself for batch_size = base_batch_size.
    """
    return base_lr * math.sqrt(batch_size / base_batch_size)

def max_cores_from_argv(default=None, argv=None):
    """Reads "--max-cores N" from the command line of a script (default if it is not given)."""
    argv = sys.argv if argv is None else argv
    if "--max-cores" in argv:
        return int(argv[argv.index("--max-cores") + 1])
    return default


##### SAVED MODELS #####
# The trained networks are written to disk (torch.save), so later tests can load them instead of training them again.
# A saved network is only reused when everything that went into its training is the same: the model class and its
# settings, the training settings, the seed, and the training and validation data themselves (a hash of the tensors).

def data_fingerprint(*tensors):
    """A short hash of the values of tensors: two data sets with the same fingerprint are the same."""
    h = hashlib.sha1()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def save_model(path, model_cls, model_kwargs, state_dict, **extra):
    """
    Saves a trained network to path (a .pt file): its class name, the arguments to rebuild it, its weights, and any
    extra information (e.g. the losses per epoch and the training settings). Load it again with load_model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save({"model_cls": model_cls.__name__, "model_kwargs": dict(model_kwargs), "state_dict": state_dict,
                "extra": extra}, path)

def load_model(path, device="cpu", model_classes=None):
    """
    Loads a network saved with save_model. Returns (model, saved), with the model in eval mode and saved the whole
    saved dictionary (model_cls, model_kwargs, state_dict, extra).
    model_classes: {name: class} for classes that are not in models.py.
    """
    import models
    classes = {"LipschitzModel": models.LipschitzModel, "UnconstrainedModel": models.UnconstrainedModel}
    classes.update(model_classes or {})
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = classes[saved["model_cls"]](**saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved


##### PARALLEL SWEEPS #####
# TRAIN_VERSION goes into the settings of saved networks and results. Raise it when a change to the training changes the
# trained networks, so that networks saved before are trained again instead of loaded.
# 2: the best weights are restored also when the num_epochs cap ends the training (before: only after early stopping).
TRAIN_VERSION = 2

# The networks here are small enough that one CPU core per model is as fast as using all of them, so independent
# trainings (e.g. a scan over Lipschitz constants, depths or widths) can run side by side in separate processes.

def fit(model_cls, model_kwargs, trn_data, val_data, test_data, lr, num_epochs, batch_size,
        max_iter=None, lr_patience=None, seed=42, eval_batch_size=4096, device="cpu", model_path=None,
        train_eval_events=50000, eval_every=1):
    # Trains a single model with Adam and MSE loss. The *_data arguments are (inputs, targets) tuples of tensors.
    # max_iter is the early stopping patience of train(), None means no early stopping.
    # lr_patience: halve the learning rate when the validation loss has not improved for more than this many epochs.
    # None means a constant learning rate.
    # model_path: a .pt file for the trained network. If it exists and was trained with the same model, settings, seed
    # and data, it is loaded instead of trained again. Otherwise the network is trained and saved there.
    # None: nothing is saved.
    # train_eval_events: the training loss is recorded on a fixed random subsample of this many training events (None:
    # all). It is only recorded, never used for a decision, so this does not change the trained network.
    # eval_every: evaluate only every this many epochs (see train()); max_iter and lr_patience then count evaluations.
    # Returns a dictionary with the per-epoch losses, the test predictions, the trained weights (to rebuild the model)
    # and "loaded" (True if the network was loaded from model_path instead of trained).
    if max_iter is None:
        max_iter = num_epochs
    test_loader = TensorBatches(*test_data, batch_size=eval_batch_size)

    settings = None
    if model_path is not None:
        settings = dict(lr=lr, num_epochs=num_epochs, batch_size=batch_size, max_iter=max_iter, lr_patience=lr_patience,
                        seed=seed, train_eval_events=train_eval_events, eval_every=eval_every,
                        train_version=TRAIN_VERSION, data=data_fingerprint(*trn_data, *val_data))
        if os.path.exists(model_path):
            try:
                model, saved = load_model(model_path, device=device, model_classes={model_cls.__name__: model_cls})
            except Exception as error: # An unreadable or old file is trained again
                print(f"Could not load {model_path} ({error}), training again")
                saved = None
            if (saved is not None and saved["model_cls"] == model_cls.__name__
                    and saved["model_kwargs"] == dict(model_kwargs) and saved["extra"].get("settings") == settings):
                return {"train_losses": saved["extra"]["train_losses"],
                        "val_losses": saved["extra"]["val_losses"],
                        "eval_epochs": saved["extra"]["eval_epochs"],
                        "best_epoch": saved["extra"]["best_epoch"],
                        "predictions": predict(model, test_loader, device),
                        "state_dict": model.state_dict(),
                        "loaded": True}

    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # The same batches as DataLoader(TensorDataset(...), shuffle=True, generator=generator), only faster (see above)
    trn_loader = TensorBatches(*trn_data, batch_size=batch_size, shuffle=True, generator=generator)
    # The training loss on a fixed subsample. Its own generator, so the training batches are not affected.
    trn_eval_data = trn_data
    if train_eval_events is not None and train_eval_events < len(trn_data[0]):
        subsample = torch.randperm(len(trn_data[0]), generator=torch.Generator().manual_seed(seed))[:train_eval_events]
        trn_eval_data = (trn_data[0][subsample], trn_data[1][subsample])
    trn_eval_loader = TensorBatches(*trn_eval_data, batch_size=eval_batch_size)
    val_loader = TensorBatches(*val_data, batch_size=eval_batch_size)

    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)

    history = {}
    train_losses, val_losses = train(model, trn_loader, val_loader, optimizer, nn.MSELoss(), num_epochs, max_iter, device,
                                     train_eval_loader=trn_eval_loader, verbose=False, scheduler=scheduler,
                                     eval_every=eval_every, history=history)

    if model_path is not None:
        save_model(model_path, model_cls, model_kwargs, model.state_dict(), settings=settings,
                   train_losses=train_losses, val_losses=val_losses, **history)

    return {"train_losses": train_losses,
            "val_losses": val_losses,
            "eval_epochs": history["eval_epochs"], # the epoch (from 1) of every entry of the losses
            "best_epoch": history["best_epoch"],   # the epoch whose weights the network ends with
            "predictions": predict(model, test_loader, device),
            "state_dict": model.state_dict(),
            "loaded": False}

def _fit_worker(args):
    job, n_threads = args
    torch.set_num_threads(n_threads)
    return fit(**job)

def fit_many(jobs, max_cores=None, n_workers=None, threads_per_worker=1):
    # Runs fit(**job) for every job in parallel and returns the results in the same order.
    # max_cores: the most cores the sweep may use (None: half the available cores, the old default).
    # n_workers: an extra cap on the number of networks trained at the same time (kept for older scripts).
    # threads_per_worker: PyTorch threads per network (1 is as fast for these small networks). The sweep uses
    # workers * threads_per_worker cores; with max_cores given the workers are reduced so that this stays within it.
    # model_cls must be importable (i.e. defined in a module, not in the notebook) to be sent to the worker processes.
    threads_per_worker = max(1, int(threads_per_worker))
    worker_cap = None if max_cores is None else max(1, max_cores // threads_per_worker)
    n_workers = n_parallel_workers(len(jobs), worker_cap, n_workers, default=lambda n_cores: max(1, n_cores // 2))

    with mp.get_context("spawn").Pool(n_workers) as pool:
        # chunksize=1: a free worker takes the next job, so jobs of different length are spread evenly over the workers
        return pool.map(_fit_worker, [(job, threads_per_worker) for job in jobs], chunksize=1)
