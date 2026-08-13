#!/bin/bash
# PRIMAL2 porting, stage 2: conda-forge env, TF1.15, od_mstar3, official models.
set -x
LOG=~/primal2_port.log
{
echo "=== $(date) stage 2 ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n primal2 -c conda-forge --override-channels python=3.7 p7zip 2>&1 | tail -3
conda activate primal2 || exit 1
python --version || exit 1
pip install -q "tensorflow==1.15" "numpy<1.19" scipy matplotlib imageio "networkx==2.5" cython 2>&1 | tail -3
cd ~/mapf-baselines/PRIMAL2/od_mstar3 && python setup.py build_ext --inplace 2>&1 | tail -3
cd ~/mapf-baselines/PRIMAL2
python -c "import tensorflow as tf; print('TF OK', tf.__version__)"
python -c "from od_mstar3 import cpp_mstar; print('od_mstar3 OK')" 2>&1 | tail -1
if [ ! -d model_primal2_oneshot ]; then
  wget -q "https://www.dropbox.com/s/3nppkpy7psg0j5v/model_PRIMAL2_oneshot_3astarMaps.7z?dl=1" -O /tmp/p2_oneshot.7z \
    && 7za x -y /tmp/p2_oneshot.7z > /dev/null && ls -d model* | head -5
fi
echo "=== inference entry inspection ==="
grep -n "saver.restore\|model_path\|load_model" driver.py parameters.py 2>/dev/null | head -8
} >> "$LOG" 2>&1
tail -25 "$LOG"
