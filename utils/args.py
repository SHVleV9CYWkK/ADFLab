# args.py
import argparse


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args_for_dataset():
    parser = argparse.ArgumentParser(description="Dataset splitting for federated learning")
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'yahooanswers', 'tiny_imagenet'],
                        help='Dataset name.')
    parser.add_argument('--clients_num', type=int, default=10, help='Number of clients.')
    parser.add_argument('--n_clusters', type=int, default=-1, help='Number of clusters when using clusters split method.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--split_method', type=str, default='train',
                        choices=['dirichlet', 'label', 'clusters', 'even'],
                        help='Non-IID splitting method: dirichlet, label, clusters, or even.')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Degree of non-IID for Dirichlet split (smaller -> more skew).')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Fraction of the dataset to use.')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Proportion of the test set.')
    parser.add_argument('--number_label', type=int, default=2,
                        help='For label split: number of label types per client.')
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='Root directory to save per-client index files.')
    return parser.parse_args()


def parse_args_for_visualization():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory to read from.')
    parser.add_argument('--save_dir', type=str, default=None, help='Optional output directory to save figures.')
    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized / Asynchronous Federated Learning")

    # ----- Core experiment selections -----
    parser.add_argument('--mode', type=str, default='async', choices=['sync', 'async'],
                        help='Execution mode: synchronous (round-based) or asynchronous (event-driven).')
    parser.add_argument('--fl_method', type=str, default='dfedavg',
                        choices=['dfedavg', 'dfedcad', 'dfedmtkd', 'dfedmtkdrl', 'dfedpgp', 'dfedsam', 'fedgo', 'qfedcg', 'retfhd', 'async_dfedavg', 'adfedmac', 'independent', 'swift', 'divshare', 'cadfedfilter'],
                        help='Decentralized FL method (local training + on-receive aggregation policy).')

    # ----- Dataset / model / optimizer -----
    parser.add_argument('--dataset_name', type=str, default='emnist',
                        choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'tiny_imagenet'],
                        help='Dataset name.')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Dataset alpha to select the split version.')
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['cnn', 'alexnet', 'leafcnn1', 'lenet', 'mobilebart', 'resnet18', 'vgg16', 'resnet50'],
                        help='Model name.')
    parser.add_argument('--optimizer_name', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer name.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Local learning rate.')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local epochs per training burst.')
    parser.add_argument('--batch_size', type=int, default=64, help='Local batch size.')
    parser.add_argument('--n_rounds', type=int, default=50,
                        help='(Sync mode) number of global rounds; kept for scheduler compatibility.')
    parser.add_argument('--scheduler_name', type=str, default='reduce_on_plateau',
                        choices=['sqrt', 'linear', 'constant', 'cosine_annealing', 'multi_step', 'reduce_on_plateau'],
                        help='Learning-rate scheduler.')

    # ----- Overlay topology (fixed; no rewiring in async_fl mode) -----
    parser.add_argument('--num_conn', type=int, default=10,
                        help='Target number of neighbors (degree if undirected; out-degree if directed).')
    parser.add_argument('--symmetry', type=int, default=0,
                        help='Non-zero -> undirected overlay; 0 -> directed overlay.')
    parser.add_argument('--gossip', type=int, default=1,
                        help='Compatibility flag with older code (not used by async_fl coordinator).')

    # ----- Asynchronous protocol specifics -----
    parser.add_argument('--total_time', type=float, default=None,
                        help='[ASYNC] Total simulation time. Defaults to --n_rounds if not set.')
    parser.add_argument('--k_push', type=int, default=None,
                        help='Max neighbors to push to at each completion event (<= num_conn). '
                             'If None, defaults to num_conn.')
    parser.add_argument('--eval_interval', type=float, default=1.0,
                        help='Wall-clock interval for evaluation ticks. Set 0 to disable periodic evaluation.')
    parser.add_argument('--compute_time_mode', type=str, default='steps*t_step',
                        choices=['constant', 'steps*t_step'],
                        help='How to compute the wall-clock duration of a local training burst.')
    parser.add_argument('--compute_interval', type=float, default=1.0,
                        help='Used when compute_time_mode=constant: fixed duration per burst.')
    parser.add_argument('--t_step', type=float, default=0.01,
                        help='Used when compute_time_mode=steps*t_step: seconds per local step.')
    parser.add_argument('--fuse_on_receive', type=_str2bool, nargs='?', const=True, default=True,
                        help='If true, fuse incoming neighbor models immediately on receipt.')
    parser.add_argument('--buffer_limit', type=int, default=16,
                        help='Max buffered neighbor updates when fuse_on_receive is false.')

    # ----- Delayed joins (Join only; no Leave) -----
    parser.add_argument('--join_mode', type=str, default='table',
                        choices=['table', 'none'],
                        help='How to specify delayed client joins. '
                             '"table": provide a JSON mapping file via --join_table; '
                             '"none": all clients join at time 0.')
    parser.add_argument('--join_table', type=str, default=None,
                        help='Path to a JSON file: { "<client_id>": <join_time_float>, ... }. '
                             'If not set and join_mode=table, all join at t=0.')

    # ----- Deprecated (kept for backward compatibility in sync_fl code paths) -----
    parser.add_argument('--delay_client_ratio', type=float, default=0.5,
                        help='[DEPRECATED for async_fl] Ratio of delayed clients (use --join_table instead).')
    parser.add_argument('--set_single_delay_client', type=int, default=-1,
                        help='[DEPRECATED for async_fl] When temp_client_dist=single, choose a client id; -1 = random.')
    parser.add_argument('--minimum_join_rounds', type=int, default=25,
                        help='[DEPRECATED for async_fl] Round to start joining new clients.')
    parser.add_argument('--temp_client_dist', type=str, default='single',
                        choices=['uniform', 'even', 'normal', 'single'],
                        help='[DEPRECATED for async_fl] Distribution of delayed clients over rounds.')

    # ----- Method-specific hyper-parameters -----
    parser.add_argument('--n_clusters', type=int, default=16,
                        help='Number of weight clusters used by DKM (if applicable).')
    parser.add_argument('--base_decay_rate', type=float, default=0.5,
                        help='Local momentum decay base rate.')
    parser.add_argument('--lambda_kd', type=float, default=0.1, help='Distillation strength.')
    parser.add_argument('--lambda_alignment', type=float, default=0.01, help='Alignment strength.')
    parser.add_argument('--lambda_feature_kd', type=float, default=0.1, help='Feature distillation strength.')
    parser.add_argument('--rho', type=float, default=0.05,
                        help='Sharpness-Aware Minimization (DFedSAM) radius.')

    # ----- Infra / logging -----
    parser.add_argument('--n_job', type=int, default=-1,
                        help='(Sync server) number of parallel training processes.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Compute device.')
    parser.add_argument('--split_method', type=str,
                        choices=['dirichlet', 'label', 'clusters', 'even'],
                        help='If provided, used to select the dataset index split.')
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='Root directory containing per-client dataset index files.')

    args = parser.parse_args()

    # Backward-compatible tweak
    if args.fl_method == 'dfedcad' and args.lambda_alignment == 0.0:
        args.fl_method = 'dfedcad_without_alignment'

    # Post-processing for async_fl defaults
    if args.k_push is None:
        args.k_push = args.num_conn

    return args