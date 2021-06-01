#!/bin/bash
#SBATCH -J ImgVGGD
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/ImgVGGD.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/ImgVGGD.err
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --mail-type=END
#SBATCH --time 10-00:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py --resume /home/mmoursi/Hiwi_Task1/saved/models/Img_VGGD/2205_102435/model_best.pth