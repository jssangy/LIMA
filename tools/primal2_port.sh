#!/bin/bash
# Attempt to resurrect PRIMAL2 (TF1-era) for inference on our instances.
# Strategy: user-space miniconda -> python 3.7 env -> tensorflow 1.15 ->
# compile od_mstar3 -> smoke inference with the shipped pretrained model.
set -x
cd ~ || exit 1
LOG=~/primal2_port.log
{
echo "=== $(date) PRIMAL2 porting attempt ==="
if [ ! -d ~/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p ~/miniconda3
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n primal2 python=3.7 2>&1 | tail -2
conda activate primal2
python --version
pip install -q "tensorflow==1.15" "numpy<1.19" scipy matplotlib imageio networkx==2.5 2>&1 | tail -3
[ -d ~/mapf-baselines/PRIMAL2 ] || git clone -q https://github.com/marmotlab/PRIMAL2.git ~/mapf-baselines/PRIMAL2
cd ~/mapf-baselines/PRIMAL2
ls
# compile the od_mstar3 cython extension
pip install -q cython 2>&1 | tail -1
cd od_mstar3 2>/dev/null && python setup.py build_ext --inplace 2>&1 | tail -3 && cd ..
python -c "import tensorflow as tf; print('TF', tf.__version__)"
ls model_primal2* 2>/dev/null || ls saved_models 2>/dev/null || find . -maxdepth 2 -name "*.ckpt*" | head -5
echo "=== smoke test placeholder: inspect inference entry points ==="
ls *.py | head -20
} > "$LOG" 2>&1
tail -30 "$LOG"
