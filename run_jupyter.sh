#!/bin/bash
#SBATCH --nodes 1
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12
#SBATCH --partition=p_general
#SBATCH --job-name jupyter-lab
#SBATCH --output jupyter.log

# ------------- NAO EXCLUIR AS LINHAS ABAIXO -----------------
JUPYTER_RUNTIME_DIR="/tmp"
XDG_RUNTIME_DIR=""
port=$(($SLURM_JOBID % 1000 + 9000))
node="$(hostname -s).sense.dcc.ufmg.br"
user=$(whoami)
export IPYTHONDIR=/local/${user}/.ipython
export JUPYTER_DATA_DIR=/local/${user}/.local/share

echo -e "

DIGITE O COMANDO ABAIXO EM UM TERMINAL NA SUA MÁQUINA PARA ABRIR O TUNEL:

ssh -N -L ${port}:${node}:${port} ${user}@slurm.sense.dcc.ufmg.br

"

# ---------- INSIRA MODULOS E SEU ENVIRONMENT AQUI -----------
conda --version
conda env list

module load cuda/10.2
module load cudnn/7.6.5_for_cuda_10.2

# ---------------------- NAO EXCLUIR -------------------------

jupyter-lab --no-browser --port=${port} --ip=${node}
