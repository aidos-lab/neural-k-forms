# Submit `MoleculeNet` data sets. In contrast to the small TU data sets,
# we just repeat the training procedure five times.

for i in `seq 5`; do
  for DATASET in BACE BBBP; do
    ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 32 --hidden-dim 64 --num-steps 10 --max-epochs 100 --tag final"
    for BASELINE in GAT GCN GIN; do
      ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --batch-size 32 --hidden-dim 64 --num-steps 10 --max-epochs 100 --tag final --baseline ${BASELINE}"
    done
  done
done
