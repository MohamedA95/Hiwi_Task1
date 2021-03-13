#!/bin/bash
#SBATCH -J QuantVGGTEST 
#SBATCH -N 1
#SBATCH -o res/QuantvggTest.out
#SBATCH -e res/QuantvggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py --resume saved/models/CIFAR_Quant_VGG/1003_215454/checkpoint-epoch25.pth

