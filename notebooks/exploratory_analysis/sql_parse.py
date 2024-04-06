import re

def sql_parse(text):
    '''
    input: text (string) - sql create table statement
    output: schema table (dictionary) - dictionary where keys are table names and values are the correponding column names respectively
    '''

    sql_split = text.split("\n")
    schema = {}

    for text in sql_split:
        table_match = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\(", text)
        if table_match and table_match.group(1) not in schema:
            table_name = table_match.group(1)
            schema[table_name] = []
        
        column_match = re.search(r"^\t(.*?),", text)
        if column_match:
            term = column_match[0].split()
            if ("PRIMARY" not in term and "KEY" not in term) or ("FOREIGN" not in term and "KEY" not in term):
                schema[table_name].append(term[0])
    return schema


if __name__ == '__main__':
    
    sql_txt = open("schema.txt", "r")
    sql_statement = sql_txt.read()
    
    print(type(sql_statement))
    print(sql_statement)
    print(sql_parse(sql_statement))