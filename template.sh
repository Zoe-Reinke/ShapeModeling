#!/bin/bash
# SLURM array template for ShapeModeling (same pattern as ShapeModelReconstruction/template.sh).
# CPU-only: ShapeWorks grooming/ICP needs no GPU. Resources are set per submission in
# submit_reference_thyroid.sh (--cpus-per-task, --mem, --time).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --output=/home/zore25/Repositories/ShapeModeling/logs/%A_%a_result.out
#SBATCH --error=/home/zore25/Repositories/ShapeModeling/logs/%A_%a_error.err

source /opt/miniconda3/etc/profile.d/conda.sh
cd /home/zore25/Repositories/ShapeModeling

# ShapeWorks lives in one of these envs -- use the first that can import it AND this package
# (shape_modeling.analysis pulls in scipy / scikit-learn / scikit-image).
SW_ENV=""
for env in /home/zore25/.conda/envs/shapemodelrecon /home/zore25/.conda/envs/shaperecon; do
    if PYTHONPATH="$PWD/src" conda run -p "$env" python -c \
        "import shapeworks, SimpleITK, shape_modeling.analysis" >/dev/null 2>&1; then
        SW_ENV="$env"; break
    fi
done
if [ -z "$SW_ENV" ]; then
    echo "no env imports shapeworks + SimpleITK + shape_modeling.analysis -- run the probe by" >&2
    echo "hand to see what is missing (see README_reference.md)" >&2; exit 1
fi
conda activate "$SW_ENV"
echo "env: $SW_ENV  shapeworks $(python -c 'import shapeworks as sw; print(getattr(sw, "__version__", "?"))')"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PYTHONPATH"
# one pair at a time per task; keep TBB/VTK/BLAS from oversubscribing the allocated cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MONITOR_LOG="logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_usage.csv"
if python -c "import psutil" 2>/dev/null && [ -f monitor_resources.py ]; then
    python3 monitor_resources.py "$MONITOR_LOG" 10 &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null" EXIT
fi

python3 -u run_job.py $SLURM_ARRAY_TASK_ID "$EXPERIMENTS_FILE"
