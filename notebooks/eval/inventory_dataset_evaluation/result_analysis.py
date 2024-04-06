import pandas as pd
import numpy as np
import re
import sys


version = sys.argv[1]
df = pd.read_csv(f'v{version}/predict.csv').fillna(0)

df['Correct'] = np.where(np.array(df['true_ans']) == np.array(df['predict_ans']), 1, 0)
correct = df['Correct'].sum()
print(f"Number of correct output: {correct}; Accuracy: {correct/50 * 100}%")
wrong_idx = df[df['Correct']==0].index
print(f'ANALYSE V{version} RESULTS')
for inspect_num in wrong_idx:
    print(' ')
    print(f'Index: {inspect_num}')
    print(f"Question: {df['question'].iloc[inspect_num]}")

    print(f"Target query: {df['query'].iloc[inspect_num]}")
    print(f"Predicted query: {df['predicted'].iloc[inspect_num]}")

    print(f"Target output: {df['true_ans'].iloc[inspect_num]}")
    print(f"Predicted output: {df['predict_ans'].iloc[inspect_num]}")

    truth =  df['true_ans'].iloc[inspect_num]
    answer = df['predict_ans'].iloc[inspect_num]

    if answer not in [0, 'error'] and truth not in [0]:
        truth = truth[1:-1]
        answer = answer[1:-1]
        segments_truth = re.findall(r'\(.*?\)', truth)
        segments_answer = re.findall(r'\(.*?\)', answer)

        print(f"Sorted and distinct target output: {sorted(list(set(segments_truth)))}")
        print(f"Sorted and distinct predicted output: {sorted(list(set(segments_answer)))}")
        
        if sorted(list(set(segments_truth))) == sorted(list(set(segments_answer))):
            print('The answer is CORRECT.')
            correct += 1
            print(f"Number of correct output: {correct}; Accuracy: {correct/50 * 100}%")
        else:
            print('The answer is WRONG.')
            print(f"Number of correct output: {correct}; Accuracy: {correct/50 * 100}%")

