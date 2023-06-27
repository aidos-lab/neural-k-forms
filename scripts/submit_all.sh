for DATASET in AIDS BZR COX2 DHFR ENZYMES Letter-high Letter-med Letter-low PROTEINS_full; do
  ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 16 --hidden-dim 16 --num-steps 5 --max-epochs 100"
  for BASELINE in GAT GCN GIN; do
    ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 16 --hidden-dim 16 --num-steps 5 --max-epochs 100 --baseline ${BASELINE}"
  done
done
