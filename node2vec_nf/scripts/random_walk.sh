#!/bin/bash

RW_JAR_FILE=/Users/doc019/Code/stellar-random-walk-research/randomwalk/target/randomwalk-0.0.1-SNAPSHOT.jar
INPUT_EDGE_LIST=nf_edge_homogeneous.txt
OUTPUT_DIR=nf_1000_rw
WALK_LENGTH=10
NUM_WALKS=10
NUM_RUNS=1
DIRECTED=false    # tested on undirected graphs only.
P=1
Q=1
SEED=132834
METHOD_TYPE=m1
WALK_TYPE=firstorder
DELIMITER="\ "    # e.g., tab-separated ("\\t"), or comma-separated (",").
LOG_PERIOD=1      # after what number of steps log the output
LOG_ERRORS=false  # Should it compute and log transition probability errors (computation intensive)
INIT_EDGE_SIZE=0.5    # portion of edges to be used to construct the initial graph
STREAM_SIZE=0.0001    # portion of edges to be used for streaming at each step
MAX_STEPS=1          # max number of steps to run the experiment
GROUPED=false         # whether the edge list is already tagged with group number (e.g., year)

# You can customize the JVM memory size by modifying -Xms.
# To run the script on the background: nohup sh random_walk.sh > log.txt &

# cmd options:
#   probs: 1st order walks
#   soProbs: 2nd order walks
#   sca: Streaming update experiment
#   degrees: Node degrees

java -Xms5g -jar $RW_JAR_FILE  --cmd probs --walkLength $WALK_LENGTH --numWalks $NUM_WALKS \
    --input $INPUT_EDGE_LIST --output $OUTPUT_DIR --nRuns $NUM_RUNS --directed $DIRECTED --p $P \
    --q $Q --seed $SEED --d "$DELIMITER" --rrType $METHOD_TYPE --wType $WALK_TYPE --save $LOG_PERIOD \
    --logErrors $LOG_ERRORS
