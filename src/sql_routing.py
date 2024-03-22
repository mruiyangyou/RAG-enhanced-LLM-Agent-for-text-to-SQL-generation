from langchain_core.runnables import RunnableParallel, RunnableLambda
from chain import * 


general_chain = rerank_chain

join_chain = (
    RunnableParallel({
        # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
        'schema': itemgetter('schema'),
        'examples': itemgetter('masked') | rerank_retriever_join | RunnableLambda(lambda x: summarise_qa_from_result(x)),
        'docs': itemgetter('masked') | rerank_retriever_join | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
        'question':itemgetter('question')
    })
    | prompt 
    | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
).with_config({"tags": ["filter_join_rag"]})


no_join_chain = (
    RunnableParallel({
        # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
        'schema': itemgetter('schema'),
        'examples': itemgetter('masked') | rerank_retriever_no_join | RunnableLambda(lambda x: summarise_qa_from_result(x)),
        'docs': itemgetter('masked') | rerank_retriever_no_join | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
        'question':itemgetter('question')
    })
    | prompt 
    | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
    
).with_config({"tags": ["filter_join_rag"]})


def route(info):
    if info['topic'] == True:
        return join_chain
    elif info['topic'] == False:
        return no_join_chain
    else:
        return join_chain
    
    
    
route_chain = RunnableLambda(route)