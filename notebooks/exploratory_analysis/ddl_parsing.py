import re

def preprocess_ddl(ddl_text):
    # Split the DDL text into chunks for each table
    # We split by "CREATE TABLE" and filter out any empty strings
    table_chunks = [chunk for chunk in ddl_text.split("CREATE TABLE") if chunk.strip()]
    
    # Adding "CREATE TABLE" back to each chunk as it was removed in the split
    table_chunks = ["CREATE TABLE" + chunk for chunk in table_chunks]
    
    return table_chunks

def find_table_schemas_with_keyword(table_chunks, keyword):
    # Search each chunk for the keyword and collect relevant chunks
    relevant_chunks = [chunk for chunk in table_chunks if keyword in chunk]
    return relevant_chunks



if __name__ == "__main__":
    
    ddl_text = '''
    CREATE TABLE author (
        aid INTEGER, 
        homepage TEXT, 
        name TEXT, 
        oid INTEGER, 
        PRIMARY KEY (aid), 
        FOREIGN KEY(oid) REFERENCES organization (oid)
    )

    CREATE TABLE cite (
        cited INTEGER, 
        citing INTEGER, 
        FOREIGN KEY(citing) REFERENCES publication (pid), 
        FOREIGN KEY(cited) REFERENCES publication (pid)
    )

    CREATE TABLE conference (
        cid INTEGER, 
        homepage TEXT, 
        name TEXT, 
        PRIMARY KEY (cid)
    )

    CREATE TABLE domain (
        did INTEGER, 
        name TEXT, 
        PRIMARY KEY (did)
    )

    CREATE TABLE domain_author (
        aid INTEGER, 
        did INTEGER, 
        PRIMARY KEY (did, aid), 
        FOREIGN KEY(did) REFERENCES domain (did), 
        FOREIGN KEY(aid) REFERENCES author (aid)
    )

    CREATE TABLE domain_conference (
        cid INTEGER, 
        did INTEGER, 
        PRIMARY KEY (did, cid), 
        FOREIGN KEY(did) REFERENCES domain (did), 
        FOREIGN KEY(cid) REFERENCES conference (cid)
    )

    CREATE TABLE domain_journal (
        did INTEGER, 
        jid INTEGER, 
        PRIMARY KEY (did, jid), 
        FOREIGN KEY(did) REFERENCES domain (did), 
        FOREIGN KEY(jid) REFERENCES journal (jid)
    )

    CREATE TABLE domain_keyword (
        did INTEGER, 
        kid INTEGER, 
        PRIMARY KEY (did, kid), 
        FOREIGN KEY(did) REFERENCES domain (did), 
        FOREIGN KEY(kid) REFERENCES keyword (kid)
    )

    CREATE TABLE domain_publication (
        did INTEGER, 
        pid INTEGER, 
        PRIMARY KEY (did, pid), 
        FOREIGN KEY(did) REFERENCES domain (did), 
        FOREIGN KEY(pid) REFERENCES publication (pid)
    )

    CREATE TABLE journal (
        homepage TEXT, 
        jid INTEGER, 
        name TEXT, 
        PRIMARY KEY (jid)
    )

    CREATE TABLE keyword (
        keyword TEXT, 
        kid INTEGER, 
        PRIMARY KEY (kid)
    )

    CREATE TABLE organization (
        continent TEXT, 
        homepage TEXT, 
        name TEXT, 
        oid INTEGER, 
        PRIMARY KEY (oid)
    )

    CREATE TABLE publication (
        abstract TEXT, 
        cid TEXT, 
        citation_num INTEGER, 
        jid INTEGER, 
        pid INTEGER, 
        reference_num INTEGER, 
        title TEXT, 
        year INTEGER, 
        PRIMARY KEY (pid), 
        FOREIGN KEY(cid) REFERENCES conference (cid), 
        FOREIGN KEY(jid) REFERENCES journal (jid)
    )

    CREATE TABLE publication_keyword (
        pid INTEGER, 
        kid INTEGER, 
        PRIMARY KEY (kid, pid), 
        FOREIGN KEY(pid) REFERENCES publication (pid), 
        FOREIGN KEY(kid) REFERENCES keyword (kid)
    )

    CREATE TABLE writes (
        aid INTEGER, 
        pid INTEGER, 
        PRIMARY KEY (aid, pid), 
        FOREIGN KEY(pid) REFERENCES publication (pid), 
        FOREIGN KEY(aid) REFERENCES author (aid)
    )
    '''

    # Preprocess the DDL text
    table_chunks = preprocess_ddl(ddl_text)

    keyword = 'homepage'

    # Search for the keyword 'homepage' and get all relevant schemas
    relevant_schemas = find_table_schemas_with_keyword(table_chunks, keyword)

    # Print the chunks containing the keyword
    for schema in relevant_schemas:
        print(schema, "\n---\n")  
