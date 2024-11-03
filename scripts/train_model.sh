set -ex
# python3 main.py --root_dir ./datasets/fundusimage1000/1000images --train --num_epochs 1000
# python3 main.py --root_dir /mnt/data/RiskIntel/ocular_imaging/datasets/1000images/ --train --num_epochs 1000
python main.py --root_dir /mnt/data/RiskIntel/ocular_imaging/datasets/ --train --num_epochs 1000

# --continue_training #  - to continue training from a checkpoint
# --train #start training from groud up
# python3 main.py --root_dir "/path/to/dataset" --train --image_size 128 --batch_size 64 --num_epochs 100 --lr_G 0.0002 --lr_D 0.0002 --lr_C 0.0002 --latent_dim 100
