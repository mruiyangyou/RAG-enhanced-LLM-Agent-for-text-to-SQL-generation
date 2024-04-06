

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Function to find if any three consecutive letters in word are in a column name
def contains_consecutive(word, column_name):
    
    for i in range(len(word) - 3):
        substring = word[i:i+4]
        if substring in column_name:
            return True
    return False

# Mask the question
def mask_question(question, db_schema, common_schema_toks = None):
    words = question.split()
    
    # Create a stemmer object
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]
    
    # Create a WordNetLemmatizer object
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word in the list
    words_lemmatize = [lemmatizer.lemmatize(word) for word in words]
    
    if type(db_schema) == dict:
        for table, cols in db_schema.items():
            cols.append(table)

            # Split column names with '_', for example, 'employee_id' will be split to 'employee' and 'id'
            schema_related_toks = [word.split('_') if '_' in word else word for word in cols]
            schema_related_toks = [item for element in schema_related_toks for item in (element if isinstance(element, list) else [element])]
            if common_schema_toks:
                schema_related_toks += common_schema_toks
            schema_related_toks = [lemmatizer.lemmatize(word.lower()) for word in schema_related_toks] # Lemmatise the schema related toks as well
            
            for tok in schema_related_toks:
                for i, word in enumerate(words_lemmatize):
                    if contains_consecutive(word.lower(), tok):
                        words[i] = '[MASK]'
                    elif word in schema_related_toks:
                        words[i] = '[MASK]'    

        return ' '.join(words)
    
    elif type(db_schema) == list:
        # Split column names with '_', for example, 'employee_id' will be split to 'employee' and 'id'
        schema_related_toks = [word.split('_') if '_' in word else word for word in db_schema]
        schema_related_toks = [item for element in schema_related_toks for item in (element if isinstance(element, list) else [element])]
        if common_schema_toks:
            schema_related_toks += common_schema_toks
        schema_related_toks = [lemmatizer.lemmatize(word.lower()) for word in schema_related_toks] # Lemmatise the schema related toks as well

        for tok in schema_related_toks:
            for i, word in enumerate(words_lemmatize):
                if contains_consecutive(word.lower(), tok):
                    words[i] = '[MASK]'
                elif word in schema_related_toks:
                    words[i] = '[MASK]'

        return ' '.join(words)

def get_schema_related_toks(row):
    schema_toks = []
    schema_toks = [table[0] for table in row['tables']]
    schema_toks += row['columns']
    return schema_toks

def mask_question_df(row, common_schema_related_toks):
    question, db_schema = row['question'], row['schema_toks']
    return mask_question(question, db_schema, common_schema_related_toks)



if __name__ == "__main__":
    common_schema_related_toks = ['student', 'course', 'department', 'age', 'course', 'ids', 'car', 'player', 'class', 'cities', 'member', 'employee']

    db_schema = {
            'department': ['id', 'name', 'num_employees', 'creation', 'budget_billions', 'head'],
            'course': ['id', 'math', 'english', 'computer_science']
            }
    
    # Example usage
    # question = "Which head's name has the substring 'Ha'? List the id and name."
    question = 'What are the name of math students are there in department computer science'
    masked_question = mask_question(question, db_schema, common_schema_related_toks)

    print(masked_question)
