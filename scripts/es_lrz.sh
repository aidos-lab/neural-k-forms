#!/bin/bash
#
# es-lrz.sh: easy submit --- submit a job quickly to the LRZ cluster
#
# This script tries to provide a 'fire and forget' solution for
# submitting CPU jobs to the cluster. The parameters may not be
# perfect, but it's probably sufficient for most cases.

NAME="$1"
CMD="$2"

if [ -z "$2" ]; then
  NAME="es_lrz.sh"
  CMD=$1
fi

if [ -z "$CMD" ]; then
  echo "Usage: $0 [NAME] COMMAND"
  echo "  Specify at least the command to run."
  exit -1
fi

sbatch -J ${NAME}                                     \
       -o "${NAME}_%j.out"                            \
       --clusters=serial                              \
       --partition=serial_std                         \
       --mail-user=bastian.rieck@helmholtz-munich.de  \
       --mail-type=end                                \
       --cpus-per-task=2                              \
       --mem=8G                                       \
       --wrap "poetry run ${CMD}"
