# Cifar10
python main.py --mode async --fl_method adflcenreg --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --lambda_reg 0.1 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method adflcenreg --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --lambda_reg 0.0 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method async_dfedavg --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method swift --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 16 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method divshare --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method independent --dataset_name cifar10 --split_method dirichlet --alpha 0.4 --model lenet --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs



# Cifar100
python main.py --mode async --fl_method adflcenreg --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --lambda_reg 0.1 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_50c.json --log_dir logs

python main.py --mode async --fl_method adflcenreg --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --lambda_reg 0.0 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_50c.json --log_dir logs

python main.py --mode async --fl_method async_dfedavg --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method swift --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 16 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method divshare --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --n_clusters 32 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs

python main.py --mode async --fl_method independent --dataset_name cifar100 --split_method dirichlet --alpha 0.4 --model resnet18 --optimizer_name adam --lr 0.001 --batch_size 32 --local_epochs 1 --device cuda --seed 42 --num_conn 10 --k_push 10 --total_time 60 --eval_interval 1 --compute_time_mode steps*t_step --t_step 0.1 --join_table ./join_tables/join_uniform_100c.json --log_dir logs