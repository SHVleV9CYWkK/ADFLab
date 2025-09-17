from __future__ import annotations

from typing import Dict, List

from clients.dfl_method_clients.async_fl.async_dfedavg_client import AsyncDFedAvgClient
from clients.dfl_method_clients.async_fl.dfedmac_client import DFedMACClient
from clients.dfl_method_clients.sync_fl.dfedavg_client import DFedAvgClient
from clients.dfl_method_clients.sync_fl.dfedcad_client import DFedCADClient
from clients.dfl_method_clients.sync_fl.dfedmtkd_client import DFedMTKDClient
from clients.dfl_method_clients.sync_fl.dfedmtkdrl_client import DFedMTKDRLClient
from clients.dfl_method_clients.sync_fl.dfedpgp_clent import DFedPGPClient
from clients.dfl_method_clients.sync_fl.dfedsam_client import DFedSAMClient
from clients.dfl_method_clients.sync_fl.fedgo_client import FedGOClient
from clients.dfl_method_clients.sync_fl.qfedcg_client import QFedCGClient
from clients.dfl_method_clients.sync_fl.retfhd_client import ReTFHDClient


def _base_hyperparams(args) -> Dict:
    """
    Assemble training + async_fl-related hyperparameters shared by all clients.
    """
    hp = {
        # training
        'optimizer_name': args.optimizer_name,
        'lr': args.lr,
        'bz': args.batch_size,
        'local_epochs': args.local_epochs,
        'n_rounds': args.n_rounds,             # kept for LR schedulers
        'scheduler_name': args.scheduler_name,

        # async_fl additions (Client base uses these; safe in sync_fl mode too)
        'compute_time_mode': getattr(args, 'compute_time_mode', 'steps*t_step'),
        'compute_interval': getattr(args, 'compute_interval', 1.0),
        't_step': getattr(args, 't_step', 0.01),
        'fuse_on_receive': getattr(args, 'fuse_on_receive', True),
        'buffer_limit': getattr(args, 'buffer_limit', 16),
    }
    return hp


def _pick_client_class(fl_type: str):
    if fl_type == "dfedavg":
        return DFedAvgClient
    if "dfedcad" in fl_type:
        return DFedCADClient
    if fl_type == "dfedmtkdrl":
        return DFedMTKDRLClient
    if fl_type == "dfedmtkd":
        return DFedMTKDClient
    if fl_type == "dfedpgp":
        return DFedPGPClient
    if fl_type == "dfedsam":
        return DFedSAMClient
    if fl_type == "fedgo":
        return FedGOClient
    if fl_type == "qfedcg":
        return QFedCGClient
    if fl_type == "retfhd":
        return ReTFHDClient
    if fl_type == "async_dfedavg":
        return AsyncDFedAvgClient
    if fl_type == "dfedmac":
        return DFedMACClient
    raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')


def create_client(num_client: int, args, dataset_index, full_dataset, device) -> List:
    """
    Instantiate per-client objects according to fl_method and inject all needed hyperparams.
    This is used in BOTH sync_fl and async_fl modes.
    """
    client_class = _pick_client_class(args.fl_method)
    train_hyperparam = _base_hyperparams(args)

    # method-specific extras
    if "dfedcad" in args.fl_method:
        train_hyperparam['lambda_kd'] = args.lambda_kd
        train_hyperparam['n_clusters'] = args.n_clusters
        train_hyperparam['lambda_alignment'] = args.lambda_alignment
        train_hyperparam['base_decay_rate'] = args.base_decay_rate

    elif args.fl_method == "dfedmtkdrl":
        train_hyperparam['lambda_kd'] = args.lambda_kd

    elif args.fl_method == "dfedmtkd":
        train_hyperparam['lambda_kd'] = args.lambda_kd

    elif args.fl_method == "dfedsam":
        train_hyperparam['rho'] = args.rho

    elif args.fl_method == "fedgo":
        train_hyperparam['lambda_kd'] = args.lambda_kd

    # guard: dfedpgp not supported in async_fl path (payload semantics differ)
    if getattr(args, "mode", "async_fl") == "async_fl" and args.fl_method == "dfedpgp":
        raise NotImplementedError(
            "dfedpgp is only supported in SYNC mode with the current async_fl coordinator. "
            "Please run with --mode sync_fl or choose another fl_method for async_fl."
        )

    clients_list = [None] * num_client
    for idx in range(num_client):
        clients_list[idx] = client_class(
            idx, dataset_index[idx], full_dataset, train_hyperparam, device
        )

    return clients_list