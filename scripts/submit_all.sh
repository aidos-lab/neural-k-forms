for DATASET in AIDS BZR COX2 DHFR PROTEINS; do
  ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --hidden-dim 128 --num-steps 10 --max-epochs 100"
done
