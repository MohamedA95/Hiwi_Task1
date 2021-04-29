#!/bin/bash
#SBATCH -J C100QvggTest
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/C100QvggTest.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/C100QvggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py -x True --resume /home/mmoursi/Hiwi_Task1/saved/models/C100_QVGG_Original/1504_091058/checkpoint-epoch100.pth

