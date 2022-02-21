#!/bin/bash
#This script traines a quant vgg16 model with 8bit using pytorch's original train script
#SBATCH -J QVGG16Nbn8bitImgNet
#SBATCH -N 1
#SBATCH --chdir /home/mmoursi/Hiwi_Task1
#SBATCH -o res/QVGG168bitImgNet_out.log
#SBATCH -e res/QVGG168bitImgNet_err.log
#SBATCH --gres=gpu:V100:4
#SBATCH --exclude=gpu[013,014,015]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --time=10-00:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
module load gcc/latest
python3 PytorchTrainScript.py -a QuantVGG --batch-size 256 --workers 16 --lr 0.01 --dist-url 'tcp://127.0.0.1:34567' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1024 /scratch/mmoursi