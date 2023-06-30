for DATASET in BACE BBBP; do
  ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 32 --hidden-dim 16 --num-steps 5 --max-epochs 100"
  for BASELINE in GAT GCN GIN; do
    ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 32 --hidden-dim 16 --num-steps 5 --max-epochs 100 --baseline ${BASELINE}"
  done
done
