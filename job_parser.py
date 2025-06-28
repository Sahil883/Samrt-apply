from pdfminer.high_level import extract_text
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import date
import pandas as pd

load_dotenv()

today = date.today()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    task="text-generation"
    
)


model = ChatHuggingFace(llm=llm)

prompt="""
You are an intelligent Job Posting parser.

Extract all the relevant information from the following job opening post. Present the output in structured JSON format with the following fields:

Job Title

Company Name

Location

Employment Type (e.g., Full-time, Part-time, Contract) (if mentioned)

Remote/Hybrid/Onsite (if mentioned)

Experience Required (e.g., 3+ years, entry-level, etc.)

Key Responsibilities (as bullet points)

Required Skills (as bullet points)

Preferred Skills (if mentioned)

Education Requirements (if mentioned)

Salary Range (if mentioned)

Application Deadline (if mentioned)

Posting Date (if mentioned)

Here's the job post text:
##input_job##
"""
def get_job_list():
    df=pd.read_csv("output.csv")

    l=[]
    for i in range(len(df)):
        prompt_with_data=prompt.replace('##input_job##',str(df.iloc[i]))
        res=model.invoke(prompt_with_data)
        l.append(res.content)
    return l






# Extract all text from the PDF


