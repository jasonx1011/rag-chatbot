from typing import Tuple, List
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.prompts import ChatPromptTemplate
from markdownify import markdownify as md
import pandas as pd
from bs4 import BeautifulSoup
from langchain.schema import Document
from datetime import datetime
import time
import pickle
from langchain.vectorstores import FAISS
import tiktoken
import openai
from collections import defaultdict

def create_df(raw_file, data_format="markdown"):
    df = pd.read_csv(raw_file)
    if "markdown" in data_format:
        df['link'] = df['link'].apply(md)
        df['title'] = df['title'].apply(md)
        df['body'] = df['body'].apply(md)
    elif "plain text" in data_format:
        # https://stackoverflow.com/questions/53189494/apply-beautifulsoup-function-to-pandas-dataframe
        df = df.applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    else:
        pass
    return df

def create_docs(content_str, metadata, splitter):
    docs = []  # List holding all the documents
    for chunk in splitter.split_text(content_str):
        docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

def create_vectorstore(db_fp, docs, embeddings):
    print(f"create new db & save: {db_fp}...")
    ts = time.time()
    vectorstore = FAISS.from_documents(
        docs, embedding=embeddings
    )
    vectorstore.save_local(db_fp)
    te = time.time()
    print(f"create new db & save: {db_fp}... runtime = {te-ts:.2f} done.")
    return vectorstore

def create_source_knowledge(vec_search_results):
    source_knowledge = "\n".join([x[0].page_content for x in vec_search_results])
    return source_knowledge
def load_vectorstore(db_fp, embeddings):
    # print(f"loading {db_fp}...")
    ts = time.time()
    vectorstore = FAISS.load_local(db_fp, embeddings=embeddings)
    te = time.time()
    # print(f"loading {db_fp}... runtime = {te-ts:.2f} done.")
    return vectorstore

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_system_promt():
    sys_promt=f"""You are a useful, friendly and respectful assistant.
    """
    return {"role": "system", "content": sys_promt}

def augment_prompt(query, source_knowledge):
    # feed into an augmented prompt
    augmented_prompt = f"""Using the knowledge base (markdown format) below, understand and summarize the content related to the query, than answer the query. 
    Limit the response within 100 english words.

    Knowledge base:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

def create_ref_md(vec_search_results):
    # references = list(set([f"[{x[0].metadata['title']}]({x[0].metadata['link']})" for x in vec_search_results]))
    references = []
    references_set = set()
    for item in vec_search_results:
        md_str = f"[{item[0].metadata['title']}]({item[0].metadata['link']})"
        if md_str not in references_set:
            references_set.add(md_str)
            references.append(md_str)
    ref_md = "  \n##### References or links: #####" + "".join(['\n* ' + ref for ref in references])
    return ref_md

def write_results_pkl(st, testset, chain):
    now = datetime.now()
    datetime_str = now.strftime("%m%d_%H%M")

    pkl_fp = f"./results_{datetime_str}.pickle"
    results = []
    for idx, query in enumerate(testset):
        ts = time.time()
        res = {"question": query, "response": chain.invoke({"question": query})}
        te = time.time()
        st.markdown(f"writing {pkl_fp}; {idx}, runtime = {te - ts:.2f}...")
        results.append(res)
    with open(pkl_fp, 'wb') as outfile:
        pickle.dump(results, outfile)
    st.markdown(f"writing {pkl_fp}... done.")
    return results


