import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def load_documents(file_path):
    """Загружает документы из текстового файла.

    Args:
        file_path (str): Путь к текстовому файлу.

    Returns:
        list: Список загруженных документов.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    """Разбивает документы на более мелкие чанки.

    Args:
        documents (list): Список документов.

    Returns:
        list: Список разбитых документов.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def create_vectorstore(docs, embeddings, vectorstore_path):
    """Создает и сохраняет векторную базу данных.

    Args:
        docs (list): Список документов для индексации.
        embeddings (OpenAIEmbeddings): Объект для создания эмбеддингов.
        vectorstore_path (str): Путь для сохранения векторной базы данных.

    Returns:
        SKLearnVectorStore: Созданная векторная база данных.
    """
    vectorstore = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_path=vectorstore_path,
    )
    return vectorstore


def load_vectorstore(embeddings, vectorstore_path):
    """Загружает существующую векторную базу данных.

    Args:
        embeddings (OpenAIEmbeddings): Объект для создания эмбеддингов.
        vectorstore_path (str): Путь к файлу векторной базы данных.

    Returns:
        SKLearnVectorStore: Загруженная векторная база данных.
    """
    if os.path.exists(vectorstore_path):
        vectorstore = SKLearnVectorStore(embedding=embeddings, persist_path=vectorstore_path)
        return vectorstore
    return None


def get_qa_chain(llm, vectorstore):
    """Создает цепочку RetrievalQA.

    Args:
        llm (ChatOpenAI): Объект LLM.
        vectorstore (SKLearnVectorStore): Векторная база данных.

    Returns:
        RetrievalQA: Цепочка RetrievalQA.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain


def main():
    """Основная функция для интерактивного взаимодействия с RAG-системой."""
    load_dotenv()

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()

    DOC_PATH = "llms_full.txt"
    VECTORSTORE_PATH = "sklearn_vectorstore.parquet"

    documents = load_documents(DOC_PATH)
    docs = split_documents(documents)

    vectorstore = load_vectorstore(embeddings, VECTORSTORE_PATH)
    if not vectorstore:
        vectorstore = create_vectorstore(docs, embeddings, VECTORSTORE_PATH)

    qa_chain = get_qa_chain(llm, vectorstore)

    print("Добро пожаловать в интерактивную RAG-систему!\nВведите 'выход' для завершения.")

    while True:
        query = input("\nВаш вопрос: ")
        if query.lower() == 'выход':
            break
        response = qa_chain.invoke({"query": query})
        print(f"Ответ: {response['result']}")


if __name__ == "__main__":
    main()
