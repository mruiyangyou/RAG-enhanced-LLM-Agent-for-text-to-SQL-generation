#!/bin/bash
GOLD=${GOLD:-classical_test.pkl}
PRED=${PRED:-out/predict.txt}
OUT_FILE=${OUT_FILE:-all_eval_results.json}
SUBSET=${SUBSET:-academic}
RESULT_PATH=${RESULT_PATH:-out/result.txt}

python3 evaluate_classical_detail.py --gold=$GOLD --pred=$PRED --out_file=$OUT_FILE --subset=$SUBSET > $RESULT_PATH
