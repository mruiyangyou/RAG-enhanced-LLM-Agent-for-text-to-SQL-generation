<!-- # Runnale use

1. in assign, you can pass lambda or fucntion
2. in paralle, just one step, use lambda(recommand) or function, multistep use ruunalelambda

# Solution
1. input
2. schema in tian'w work become list of dictionary(a)
3. a and input in luo's
    -->

## Bug fix

cohere_rank.py add results

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
        "evaluation/out/test/v4/predict.txt" \
        "schema"
```

### eval_test_suite
```bash
python eval_test_suite.py "rag" "evaluation" "evaluation/out/test/v6/predict.csv"
```
<!-- 
RAG-enhanced LLM Agent for text to SQL geneeation

find tuned llm for statistics question answering

machine learning system for football prediction -->


## Prompt template

### Template 1

```python
"""
You are an assistant for giving text to sql task.
Based on the table schema, relevant SQL documentation related to the question where you can learnt some functions, relevant example pairs of question-answer for SQL below
- database schema: {schema} 
- sql documentation: {docs} 
- sql simdilar examples: {examples} 

Write a SQL query that would answer the user's question.

Question: {question} 
SQL Query:
"""
```

### Template 2

```python
"""
You are tasked with converting natural language questions into SQL queries based on a provided database schema. Your response should accurately reflect the question's intent by leveraging the correct tables, columns, and SQL functions. Please follow the guidelines below: 

Step 1, you must understand the database schema. 
The database schema defines the structure of the database. Familiarize yourself with the tables and their relationships. Here's the schema: {schema}

Step 2, you must fully understand the unfamiliar terms in the question: {unfamiliar}

Step 3, you can refer to this related SQL documentation. Here’re key SQL concepts and operations for your reference: {docs}

Step 4, you can refer to similar question and SQL query examples.  Here’re example SQL queries to illustrate how specific questions can be answered using SQL: {examples}

Step 5, you must construct your SQL query. 

Given the question, your task is to construct an SQL query that retrieves the requested information from the database. Remember to: 
Identify Relevant Tables: Determine which tables contain the data needed to answer the question. 
Select the Correct Columns: Make sure to select only the columns that are necessary to answer the question. 
Construct the SQL query using appropriate SQL operations and conditions. 

Your task: Convert the given text question into an SQL query using the provided schema and documentation. Here's the question you need to answer: 

Question: {question}
SQL Query:
"""
```

## Template 3

```python

"""
Your goal is to translate natural language questions into SQL queries using a specific database schema. Follow these streamlined steps for an accurate output: 

1.Understand the Schema. Familiarize yourself with the database schema to know where data resides. 

Database schema:{schema}


2.SQL Documentation. Use provided SQL documentation for syntax and function guidance. 

SQL documentation: {docs}

 
4.Review Examples. Check similar question-query examples for insights. 

Similar examples: {examples}

 
5.Construct the Query. Identify relevant tables and columns. Craft your SQL query focusing on efficiency and accuracy. 

 
Your task: 

Transform the following question into an SQL query based on the schema and resources provided. 

Question: {question}
SQL Query: [Your SQL query here] 
"""
```

## Tempalte 4
```python
"""
Given a natural language question {question}, you are tasked with converting this question into a SQL query based on a provided database schema. Your response should accurately reflect the question's intent by leveraging the correct tables, columns, and SQL functions and return exactly the information requested, without any superfluous data. Please follow the guidelines below: 

Step 1, you must understand the database schema. 
The database schema defines the structure of the database. Familiarize yourself with the tables and their relationships. Here's the schema: {schema}

Step 2, you must fully understand the unfamiliar terms in the question. Here’re the unfamiliar terms: {unfamiliar}

Step 3, you can refer to this related SQL documentation. Here’re key SQL concepts and operations for your reference: {docs}

Step 4, you can refer to similar question and SQL query examples.  Here’re example SQL queries to illustrate how specific questions can be answered using SQL: {examples}

Step 5, you must construct your SQL query. 

Given the question, your task is to construct an SQL query that retrieves the requested information from the database. Remember to: 
Identify Relevant Tables: Determine which tables contain the data needed to answer the question. 
Select the Correct Columns: Make sure to select only the columns that are necessary to answer the question. 
Construct the SQL query using appropriate SQL operations and conditions. 
Check the query that it should return exactly the information requested, without any superfluous data. 

Your task: Convert the given text question into an SQL query using the provided schema and related information. Here's the question you need to answer: 

Question: {question}
SQL Query: 
"""
```