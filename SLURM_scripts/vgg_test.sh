#!/bin/bash
#SBATCH -J vggTest
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/vggTest.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/vggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=8192
#SBATCH --mail-type=END
date
echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py --resume /home/mmoursi/Hiwi_Task1/saved/models/C100_QVGG_Original/1504_091058/model_best.pth

