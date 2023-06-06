for DATASET in AIDS BZR COX2 DHFR ENZYMES PROTEINS; do
  ./es_lrz.sh $DATASET "python -m cochain_representation_learning.graphs --name ${DATASET} --max-epochs 100"
done
