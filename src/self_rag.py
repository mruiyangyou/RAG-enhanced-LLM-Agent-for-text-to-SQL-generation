from typing import Dict, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph import StateGraph
from chain import *
from sqlalchemy import create_engine

class GraphState(TypedDict):
    
    keys: Dict[str, any]
    
# graph

# db_path = '/Users/marceloyou/Desktop/UCL-DSML/COMP0087-Boss/SQLess/src/evaluation/database/academic/academicv1223round10group3.sqlite'   
db_path = '/Users/marceloyou/Desktop/UCL-DSML/COMP0087-Boss/SQLess/data/inventory/inventory.sqlite'
engine = create_engine(f'sqlite:///{db_path}')
db = connect_db('engine', engine = engine)
    
def exec(code: str) -> None:
    # result = db.run(code)
    # if result == '':
    #     raise ValueError('The execution did not produce any output. Please check your used table and use correct sql functions!')
    # No need to call db.run(code) again
    db.run(code)


def generate(state: GraphState):
    
    # state
    state_dict = state['keys']

    question = state_dict['question']
    iter = state_dict['iterations']
    schema = state_dict['schema']
    masked = state_dict['masked']
    
    if "error" in state_dict:
        print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

        error = state_dict["error"]
        code_solution = state_dict['generation']
        
        addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:  
                    \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code 
                    execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this. Please do not output same answer again!"""
                    
        template_refine = addendum + template
      
        prompt_update = ChatPromptTemplate.from_messages(
            [
                ("human", template_refine),
            ]
        )
        
        sql_refine_chain = (
            RunnableParallel({
                # 'schema': lambda x: get_useful_schema({'question': x['question'], 'schema': format_schema(x['schema'])}),
                'schema': itemgetter('schema'),
                'examples': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: summarise_qa_from_result(x)),
                'docs': itemgetter('masked') | rerank_retriever | RunnableLambda(lambda x: search_in_document(docs, summarise_keywords_from_result(x))),
                'question':itemgetter('question'),
                'generation': itemgetter('generation'),
                'error': itemgetter('error')
            })
            | prompt_update
            | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
            
        )  
        code_solution = sql_refine_chain.invoke(
            {'question': question, 'schema': schema, 'masked': masked, 
              'generation': str(code_solution), 'error': error}
        )
                
    else:
        print("---SQL Generation---")
        
        code_solution = rerank_chain.invoke({'question': question, 'schema': schema, 'masked': masked})
        print(code_solution)

    iter = iter + 1
    return {
        "keys": {'generation': code_solution, 'question': question, 'iterations': iter, 'masked': masked, 'schema': schema}
    }
    
    
def check_code_execution(state: GraphState):
    print("---CHECKING CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    schema = state_dict['schema']
    masked = state_dict['masked']
    
    iter = state_dict['iterations']
    
    try:
        exec(code_solution)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"
        
    return {
        "keys": {
            'generation': code_solution,
            'question': question,
            'schema': schema,
            'masked': masked,
            'error': error,
            'iterations': iter 
        }
    }
    
## edge
def decide_to_finish(state: GraphState):
    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter == 4:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"



## construction
workflow = StateGraph(GraphState)

# define the nodes

workflow.add_node('generate', generate)
workflow.add_node('check_code_execution', check_code_execution)

# build graph
workflow.set_entry_point('generate')
workflow.add_edge('generate', 'check_code_execution')

workflow.add_conditional_edges(
    "check_code_execution",
    decide_to_finish,
    {
        "end": END,
        "generate": 'generate'
    }
)

app = workflow.compile()


def model(dict: Dict):
    return app.invoke({"keys": {**dict, 'iterations': 0}})['keys']['generation']