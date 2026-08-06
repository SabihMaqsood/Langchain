from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Structured Output Schema

class ReviewAnalysis(BaseModel):

    sentiment: str = Field(description="Positive, Negative or Neutral")

    rating: int = Field(description="Rating from 1 to 5")

    summary: str = Field(description="Short review summary")

    suggestion: str = Field(description="Suggestion for improvement")


# Output Parser

parser = PydanticOutputParser(
    pydantic_object=ReviewAnalysis
)


# Static Prompt System message

system_prompt = """

You are an expert Product Review Analyzer.
Always analyze reviews carefully.
Return the output only in the requested format.

"""


# Dynamic Prompt

prompt = PromptTemplate(

    template="""
Review:

{review}

{format_instruction}
""",

    input_variables=["review"],

    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# User Input


review = input("Enter Product Review: ")


# Create Messages


messages = [

    SystemMessage(content=system_prompt),

    HumanMessage(content=prompt.format(review=review))

]


# Call Gemini

response = model.invoke(messages)


# Parse Output


result = parser.parse(response.content)

# Display Output

print("Sentiment :", result.sentiment)

print("Rating    :", result.rating)

print("Summary   :", result.summary)

print("Suggestion:", result.suggestion)