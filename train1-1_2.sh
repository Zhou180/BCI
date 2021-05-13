wandb login d7a852a8acf46e50c9ace763c15f3c75f6a794e9
wandb enabled
# nets=( EEGNet ShallowConvNet SCCNet TSception )
nets=( EEGNet )
# subject_list=( S01 S02 S03 S04 S05 S06 S07 S08 S09 )
subject_list=( S01 )
# method=( Single X X_finetune X_mix)
methods=( Single )
lr_list=( 3e-3 1e-3 3e-4 1e-4 )
batch_list=( 2 8 32 64 )
# shuffle=( True False )
shuffle_list=( True )



for net in  "${nets[@]}"
do
    for subject in "${subject_list[@]}"
    do
        for method in "${methods[@]}"
        do
            for lr_rate in "${lr_list[@]}"
            do
                for batch in "${batch_list[@]}"
                do
                    for shuffle in "${shuffle_list[@]}"   
                    do 
                        python main.py \
                        --name ${net}_${subject}_${method}_lr${lr_rate}_b${batch}_sh${shuffle} \
                        --net ${net} \
                        --subject ${subject} \
                        --method ${method} \
                        --learning_rate ${lr_rate} \
                        --batch ${batch} \
                        --shuffle ${shuffle} \
                        --epoch 100 \
                        --finetune_epoch 30 \
                        --gpu 0 \
                        --tags ${net} ${subject} ${method} batch_${batch} lr_${lr_rate} ${shuffle}
                    done
                done
            done
        done
    done
done