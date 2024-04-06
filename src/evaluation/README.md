# Evaluation commands

Steps:
1. Make sure you install docker on your machines.
2. Dowonload [database](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view) to your folder at `./evaluation/database`
3. You have `classical_test_gold.txt` at `./src/evaluation/examples`

Change to evaluation directory

```bash
cd src/evaluation
```

Docker build
```bash
docker build -t sql_test:v1 .
```

<!-- Docker run
```bash
docker run -it \
  -v $(pwd)/database:/app/database \
  -v $(pwd)/out:/app/out \
  sql_test:v1
``` -->

Docker run for all database
```bash
docker run -it \
    -e PRED="out/xxx/predict.txt" -e OUT_FILE="out/xxx/result.csv" -e SUBSET="academic" -e RESULT_PATH="out/xxx/result.txt" \
    -v $(pwd)/database:/app/database \
    -v $(pwd)/out:/app/out \
    sql_test:v1

```


Docker run sampled academic database
```bash
docker run -it \                                          
    -e GOLD="academic_test.pkl" -e PRED="out/xxx/predict.txt" -e OUT_FILE="out/xxx/result.csv" -e SUBSET="academic" -e RESULT_PATH="out/xxx/result.txt" \
    -v $(pwd)/database:/app/database \
    -v $(pwd)/out:/app/out \
    sql_test:v2
```