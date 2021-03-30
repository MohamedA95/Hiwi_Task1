#!/bin/bash
#SBATCH -J VGGTEST
#SBATCH -N 1
#SBATCH -o res/vggTest.out
#SBATCH -e res/vggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py --resume saved/models/CIFAR_VGG/0303_201026/model_best.pth

