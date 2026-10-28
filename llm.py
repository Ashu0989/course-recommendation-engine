from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = f"""
You are a university requirements analyzer. From these DataFrame columns,
identify those relevant to determining student eligibility:
{df.columns.tolist()}

Exclude unrelated fields. Use only column names that actually appear in the list.
Group the relevant columns into the mentioned categories programme_eligibility, academic requirements,
language proficiency, work experience and additional_notes only. Return a JSON object whose keys are
category names and whose values are lists of column names.
"""
response = llm.invoke(prompt)
print("Response from LLM:")
print(response.content)