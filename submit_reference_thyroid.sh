#!/bin/bash
# Submit the thyroid grooming-reference medoid computation as a dependent chain:
#   prepare (task 0) -> selfcheck (task 1) -> 200 pair chunks (tasks 100-299) -> reduce (task 2)
# Each step starts only if the previous one succeeded (afterok). The pair chunks start only after
# the self-check passed, i.e. once our pair distance is verified to reproduce ShapeWorks.
#
# Run on the login node from the repo root:  bash submit_reference_thyroid.sh
# Print the sbatch calls without submitting: DRY=1 bash submit_reference_thyroid.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

EXP=experiments/reference_thyroid.json

submit() {  # submit <args...> -> prints the job id
    if [ "${DRY:-0}" = "1" ]; then
        echo "sbatch --export=ALL,EXPERIMENTS_FILE=$EXP $*" >&2
        echo "<id:$(echo "$*" | sed -E 's/.*--job-name=([^ ]+).*/\1/')>"
    else
        sbatch --parsable --export=ALL,EXPERIMENTS_FILE="$EXP" "$@"
    fi
}

PREP=$(submit --job-name=ref_prepare --array=0 --cpus-per-task=2 --mem=8G --time=04:00:00 template.sh)
CHECK=$(submit --job-name=ref_selfcheck --array=1 --cpus-per-task=8 --mem=8G --time=02:00:00 \
    --dependency=afterok:"$PREP" template.sh)
PAIRS=$(submit --job-name=ref_pairs --array=100-299%50 --cpus-per-task=1 --mem=4G --time=04:00:00 \
    --dependency=afterok:"$CHECK" template.sh)
REDUCE=$(submit --job-name=ref_reduce --array=2 --cpus-per-task=1 --mem=4G --time=00:30:00 \
    --dependency=afterok:"$PAIRS" template.sh)

echo "prepare=$PREP selfcheck=$CHECK pairs=$PAIRS reduce=$REDUCE"
echo "results: reference_thyroid_512p_excl/{selfcheck.json,reference.json,ranking.csv}"
