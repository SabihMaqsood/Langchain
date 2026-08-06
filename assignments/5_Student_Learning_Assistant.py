import os
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

#Model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
)

# structured output class

class LearningReport(BaseModel):
    topic: str = Field(description="Main topic name")

    difficulty: str = Field(
        description="Beginner, Intermediate, or Advanced"
    )

    summary: str = Field(
        description="Short summary of the topic"
    )

    detailed_answer: str = Field(
        description="Detailed explanation"
    )

    key_points: List[str] = Field(
        description="Important learning points"
    )

    study_tips: List[str] = Field(
        description="Helpful study tips"
    )

    quiz_questions: List[str] = Field(
        description="Three quiz questions"
    )

    references: List[str] = Field(
        description="Recommended learning resources"
    )


# Structured Model
structured_model = model.with_structured_output(LearningReport)

# Prompts templates

STATIC_PROMPT = """
You are a Smart Student Learning Assistant.

Rules:
- Provide accurate academic information.
- Do not make up facts.
- Keep explanations educational.
- Identify the topic correctly.
- Determine suitable difficulty level.
- Generate useful quiz questions.
- Generate study tips.
- Return all fields properly.
"""


STUDENT_PROMPT = """
Role: Student

Instructions:
- Explain in simple language.
- Use beginner-friendly examples.
- Avoid complex terminology.
- Make learning easy.
- Give practical study tips.
"""

TEACHER_PROMPT = """
Role: Teacher

Instructions:
- Provide detailed explanation.
- Include learning objectives.
- Use classroom examples.
- Suggest teaching strategies.
- Suggest homework ideas.
"""

print("\n========== SMART STUDENT LEARNING ASSISTANT ==========\n")

print("Select Role:")
print("1. Student")
print("2. Teacher")

role_choice = input("\nEnter choice (1/2): ").strip()

if role_choice == "1":
    role_prompt = STUDENT_PROMPT
    role_name = "Student"

elif role_choice == "2":
    role_prompt = TEACHER_PROMPT
    role_name = "Teacher"

else:
    print("Invalid choice. Default role = Student")
    role_prompt = STUDENT_PROMPT
    role_name = "Student"


# USER QUESTION

question = input("\nEnter Academic Question: ")


# CREATE MESSAGES

messages = [
    SystemMessage(
        content=f"{STATIC_PROMPT}\n\n{role_prompt}"
    ),

    HumanMessage(
        content=question
    )
]

response = structured_model.invoke(messages)


parser = StrOutputParser()

formatted_text = f"""
Topic: {response.topic}

Difficulty: {response.difficulty}

Summary:
{response.summary}

Detailed Answer:
{response.detailed_answer}
"""

parsed_output = parser.invoke(formatted_text)

# DISPLAY RESULT

print("\n")
print("=" * 60)
print("SMART LEARNING REPORT")
print("=" * 60)

print(f"\nRole: {role_name}")

print("\nTOPIC")
print("-" * 30)
print(response.topic)

print("\nDIFFICULTY")
print("-" * 30)
print(response.difficulty)

print("\nSUMMARY")
print("-" * 30)
print(response.summary)

print("\nDETAILED ANSWER")
print("-" * 30)
print(response.detailed_answer)

print("\nKEY POINTS")
print("-" * 30)

for i, point in enumerate(response.key_points, start=1):
    print(f"{i}. {point}")

print("\nSTUDY TIPS")
print("-" * 30)

for i, tip in enumerate(response.study_tips, start=1):
    print(f"{i}. {tip}")

print("\nQUIZ QUESTIONS")
print("-" * 30)

for i, quiz in enumerate(response.quiz_questions, start=1):
    print(f"{i}. {quiz}")

print("\nREFERENCES")
print("-" * 30)

for i, ref in enumerate(response.references, start=1):
    print(f"{i}. {ref}")

print("\nPARSER OUTPUT")
print("-" * 30)
print(parsed_output)

print("\n" + "=" * 60)
print("END OF REPORT")
print("=" * 60)