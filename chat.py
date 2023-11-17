from collections import deque
import streamlit as st
import time
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from utils import create_df, create_docs, load_vectorstore, create_source_knowledge, \
    augment_prompt, create_ref_md, num_tokens_from_string, create_system_promt

st.title("RAG chatbot demo")

with st.sidebar:
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    if not openai_api_key:
        st.warning('Please input a valid OpenAI API Key.')
        st.stop()

    cleared = st.button("Clear conversation")
    if cleared:
        if "messages" in st.session_state:
            st.session_state.messages = []

    openai_model_name = st.radio(
        "openai_model_name",
        ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4"],
        index=0,
    )
    raw_file = st.text_input('Input csv file', './assignment_articles_cust.csv')
    # load_db = st.radio('load db', ['true', "false"], index=0)
    embedding_model_id = st.radio(
        "embedding_model_id",
        ["openai_emb", "BAAI/bge-small-en-v1.5"],
        index=0,
    )
    postfix = embedding_model_id.split("/")[-1]
    db_fp = f"./faiss_index_{postfix}"

    # st.input('vec db path', db_fp, language="markdown")
    st.code(f"{db_fp}", language="markdown")
    top_k = st.radio(
        "top_k",
        ["3", "5", "10"],
        index=0,
    )
    top_k = int(top_k)
    score_threshold = st.radio(
        "score_threshold",
        ["0.5", "0.7", "0.9"],
        index=0,
    )
    score_threshold = float(score_threshold)

client = OpenAI(api_key=openai_api_key)

# debug
expander = st.expander("Debug detail")
df = create_df(raw_file)
expander.write(df)

# select embedding model
if embedding_model_id == "openai_emb":
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = load_vectorstore(db_fp, embeddings)
else:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
    )
    vectorstore = load_vectorstore(db_fp, embeddings)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = openai_model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "post_info" in message:
            st.markdown(message["content"] + "".join(message["post_info"]))
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    vec_search_results = vectorstore.similarity_search_with_score(prompt, k=top_k, score_threshold=score_threshold)
    source_knowledge = create_source_knowledge(vec_search_results)

    post_info = []
    if vec_search_results:
        aug_content = augment_prompt(prompt, source_knowledge)
        post_info.append(create_ref_md(vec_search_results))
    else:
        aug_content = prompt

    # debug
    expander.write(f"vec search results len() = {len(vec_search_results)}")
    expander.write(vec_search_results)

    st.session_state.messages.append(
        {
            "role": "user", "content": prompt, "aug_content": aug_content,
            "content_tok_num": num_tokens_from_string(prompt, openai_model_name),
            "aug_content_tok_num": num_tokens_from_string(aug_content, openai_model_name),
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        model_messages = deque()
        token_limit = 3600
        acc_token_len = 0
        # pick history from backward and form LLM inputs
        for m in reversed(st.session_state.messages):
            if m["role"] == "user":
                acc_token_len += m["aug_content_tok_num"]
                model_messages.appendleft({"role": m["role"], "content": m["aug_content"]})
            else:
                acc_token_len += m["content_tok_num"]
                model_messages.appendleft({"role": m["role"], "content": m["content"]})
            if acc_token_len > token_limit:
                model_messages.popleft()
                break
        model_messages = [create_system_promt()] + list(model_messages)

        # # debug
        # with open("model_messages.log", "w") as outfile:
        #     outfile.write("\n".join([x['role'] + ": " + x['content'] for x in model_messages]))
        #     outfile.write(f"\nacc_token_len = {acc_token_len}")
        # st.write(f"acc_token_len = {acc_token_len}")

        ts = time.time()
        # get responses and using stream to form the chatbot output
        for response in client.chat.completions.create(
                messages=list(model_messages),
                model=st.session_state["openai_model"], stream=True,):
            if response.choices[0].delta.content:
                full_response += response.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        te = time.time()

        post_info.append(f"  \n(runtime = {te-ts:.2f} seconds)")
        message_placeholder.markdown(full_response + "".join(post_info))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "content_tok_num": num_tokens_from_string(prompt, openai_model_name),
            "post_info": post_info,
        }
    )