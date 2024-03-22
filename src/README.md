<!-- # Runnale use

1. in assign, you can pass lambda or fucntion
2. in paralle, just one step, use lambda(recommand) or function, multistep use ruunalelambda

# Solution
1. input
2. schema in tian'w work become list of dictionary(a)
3. a and input in luo's
    -->

## Evaluation Scripts


### eval_custom
```bash
python eval_custom.py "rag" "../data/inventory/inventory.csv" \
        "../data/inventory/inventory.sqlite" "evaluation/out/test/predict.csv"
```

### eval_academic
```bash
python eval_academic.py "rag" \
        "../data/sql_test_suite_academic/masked_academic_sample.csv" \
        "evaluation/out/test/v4/predict.csv" \
        "schema"
```

### eval_test_suite
```bash
python eval_test_suite.py "rag" "evaluation" "evaluation/out/test/v6/predict.csv"
```
<!-- 
RAG-enhanced llm agent for sql geneeation

find tuned llm for statistics question answering

machine learning system for football prediction -->