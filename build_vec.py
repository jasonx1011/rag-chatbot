import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from utils import create_vectorstore, create_df, create_docs


def main():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument('-m', '--embedding_model_id', dest="embedding_model_id",
                        required=True, type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument('-k', '--openai_api_key', dest="openai_api_key",
                        required=False, type=str)
    parser.add_argument('-r', '--raw_file', dest="raw_file",
                        required=False, type=str, default="./assignment_articles_cust.csv")
    parser.add_argument('-s', '--chunk_size', dest="chunk_size",
                        required=False, type=int, default=1500)
    parser.add_argument('-ol', '--chunk_overlap', dest="chunk_overlap",
                        required=False, type=int, default=0)

    args = parser.parse_args()
    embedding_model_id = args.embedding_model_id
    openai_api_key = args.openai_api_key
    # raw_file = "./assignment_articles_cust.csv"
    # chunk_size = 1500
    # chunk_overlap = 0
    # embedding_model_id = "openai_emb"
    # # embedding_model_id = "BAAI/bge-small-en-v1.5"

    raw_file = args.raw_file
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    postfix = embedding_model_id.split("/")[-1]
    db_fp = f"./faiss_index_{postfix}"
    data_format = "markdown"
    separators = ["\n\n", "\n", " ", ""]

    df = create_df(raw_file, data_format=data_format)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=separators)
    # ref: https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
    docs = []
    for idx, row in df.iterrows():
        docs.extend(create_docs(content_str=row['body'], metadata={'link': row['link'], 'title': row['title']},
                                splitter=splitter))

    # select embedding model
    if embedding_model_id == "openai_emb":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_id,
        )

    vectorstore = create_vectorstore(db_fp, docs, embeddings)


if __name__ == "__main__":
    main()
