# Submission script for the final runs of all TU data sets. The idea is
# that we run this for all five folds and collate the results.

for FOLD in 0 1 2 3 4; do
  for DATASET in AIDS BZR COX2 DHFR ENZYMES Letter-high Letter-med Letter-low PROTEINS_full; do
    ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --fold ${FOLD} --batch-size 16 --hidden-dim 16 --num-steps 5 --max-epochs 100 --tag final"
    for BASELINE in GAT GCN GIN; do
      ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --fold ${FOLD} --batch-size 16 --hidden-dim 16 --num-steps 5 --max-epochs 100 --tag final --baseline ${BASELINE}"
    done
  done
done
