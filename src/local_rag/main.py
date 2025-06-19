"""
Embedding generation intro.
"""

import boto3
from langchain_aws import BedrockLLM as Bedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


session = boto3.Session(profile_name="personal")
bedrock_runtime = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


model = Bedrock(
    model_id="amazon.titan-text-express-v1",
    client=bedrock_runtime,
)


my_data = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Andres likes to eat pizza",
    "Andres likes to eat pinto",
    "Andres dont like to eat coliflor",
]


bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_runtime,
)


def invoke_model(prompt: str):
    """
    Invoke the Bedrock model with the given prompt.
    """
    response = model.invoke(prompt)
    return response


def first_chain():
    """
    Create a simple chat prompt template.
    """
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "write a short descripton for the product provided by the user"),
            ("human", "{product_name}"),
        ]
    )
    chain = template.pipe(model)

    response = chain.invoke({"product_name": "iPhone 16"})
    print("Response: ", response)


def local_rag_chain():
    """
    Create a local RAG chain.
    """
    question = "What does Andres likes to eat?"
    vector_store = FAISS.from_texts(
        texts=my_data,
        embedding=bedrock_embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    results = retriever.get_relevant_documents(question)
    results_text = [result.page_content for result in results]
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the users question based on provided context: {context}",
            ),
            ("user", "{input}"),
        ]
    )
    chain = template.pipe(model)
    response = chain.invoke(
        {
            "input": question,
            "context": results_text,
        }
    )
    print("Response: ", response)


def local_rag_chain_pdf():
    """
    Create a local RAG chain with PDF.
    """
    loader = PyPDFLoader("src/assets/dnd_players_guide.pdf")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = loader.load()
    splitted_documents = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(
        documents=splitted_documents,
        embedding=bedrock_embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    question = "What is the main topic of the document?"
    results = retriever.get_relevant_documents(question)
    results_text = [result.page_content for result in results]

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the users question based on provided context: {context}",
            ),
            ("user", "{input}"),
        ]
    )

    chain = template.pipe(model)
    response = chain.invoke(
        {
            "input": question,
            "context": results_text,
        }
    )

    print("Response: ", response)


if __name__ == "__main__":
    print("Hello, Bedrock!")
    # prompt = "What is the highest mountain in the world?"
    # response = invoke_model(prompt)
    # print("Response: ", response)
    # first_chain()
    # local_rag_chain()
    local_rag_chain_pdf()
