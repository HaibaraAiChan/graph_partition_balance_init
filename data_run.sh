#!/bin/bash
File=pseudo_reddit_gp_w.py
# Data=(karate cora reddit ogbn-products ogbn-mag)
# Data=(reddit ogbn-products ogbn-mag)
# data=cora
Data=(ogbn-products ogbn-mag)
Aggre=mean
alpha_list=(1.0 0.5 0.2) 
# mkdir logs
for data in ${Data[@]}
do
    for alpha in ${alpha_list[@]}
    do
        python $File \
            --dataset $data \
            --aggre $Aggre \
            --selection-method balanced_init_graph_partition \
            --balanced_init_ratio 0.9 \
            --alpha $alpha \
            --num-epochs 6 \
            --eval-every 5 &> logs/${data}_${Aggre}_alpha_${alpha}_pseudo_balance_init_ratio_0.9.log
            
        python $File \
            --dataset $data \
            --aggre $Aggre \
            --selection-method balanced_init_graph_partition \
            --balanced_init_ratio 0.5 \
            --alpha $alpha \
            --num-epochs 6 \
            --eval-every 5 &> logs/${data}_${Aggre}_alpha_${alpha}_pseudo_balance_init_ratio_0.5.log
            
        python $File \
            --dataset $data \
            --aggre $Aggre \
            --selection-method balanced_init_graph_partition \
            --balanced_init_ratio 0.2 \
            --alpha $alpha \
            --num-epochs 6 \
            --eval-every 5 &> logs/${data}_${Aggre}_alpha_${alpha}_pseudo_balance_init_ratio_0.2.log
            
        python $File \
            --dataset $data \
            --aggre $Aggre \
            --selection-method random_init_graph_partition \
            --alpha ${alpha} \
            --num-epochs 6 \
            --eval-every 5 &> logs/${data}_${Aggre}_alpha_${alpha}_pseudo_random_init.log

    done
done
