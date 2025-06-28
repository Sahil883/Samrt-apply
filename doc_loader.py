from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from job_parser import get_job_list


# Load the model

job_list=get_job_list()
print(job_list)
exit()
model_name = "sentence-transformers/all-mpnet-base-v2"

loader=CSVLoader(file_path='output1.csv')

data=loader.load()


embedding_model = HuggingFaceEmbeddings(model_name=model_name)
vector_store = FAISS.from_documents(data,embedding_model)

# vector_store = FAISS.from_documents(sentence_embedding, data)
retriever = vector_store.as_retriever(    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.1, "k": 4})

res=retriever.invoke("Data Engineering jobs in Nike Company")
print(len(res))
# docs_and_scores = vector_store.similarity_search_with_score("AutoZone", k=4)
# for doc, score in docs_and_scores:
#     print(f"Score: {score:.3f}")
#     print(doc.page_content)
#     print("-----------")
