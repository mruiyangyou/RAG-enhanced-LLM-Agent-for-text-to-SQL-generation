FROM python:3.7-slim

RUN pip install sqlparse nltk pandas

WORKDIR /app

COPY ./examples/classical_test_gold.txt ./examples/classical_test_gold.txt

# COPY ./out/predict.txt ./out/predict.txt

COPY ./*.py ./

# COPY classical_test.pkl ./

# COPY academic_test.pkl ./

COPY ./*.pkl ./

COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]


# CMD ["sh", "-c", "python3 evaluate_classical.py --pred=out/predict.txt --out_file=all_eval_results.json --subset=academic > out/result.txt"]
# CMD ["sh", "-c", "python3 evaluate_classical.py --pred=out/predict.txt --out_file=all_eval_results.json --subset=academic > out/result.txt"]

# docker run -v $(pwd)/result:/app/out my-python-script


