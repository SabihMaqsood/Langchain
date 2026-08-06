# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from typing import TypedDict, Annotated, Optional, Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)



# model = ChatGoogleGenerativeAI(model="gemini-3.4-flash")


class HospitalReview(TypedDict):

    summary: Annotated[
        str,
        "Short summary of the review"
    ]

    sentiment: Annotated[
        Literal["Positive", "Negative", "Neutral"],
        "Overall sentiment"
    ]

    treatment_quality: Annotated[
        Optional[str],
        "Quality of treatment"
    ]

    doctor_behavior: Annotated[
        Optional[str],
        "Behavior of doctors"
    ]

    staff_service: Annotated[
        Optional[str],
        "Quality of hospital staff service"
    ]

    cleanliness: Annotated[
        Optional[str],
        "Hospital cleanliness"
    ]

    health_improvement: Annotated[
        Optional[str],
        "Patient health improvement after treatment"
    ]

    recommend_hospital: Annotated[
        Literal["Yes", "No", "Maybe"],
        "Would patient recommend the hospital?"
    ]

    reviewer_name: Annotated[
        Optional[str],
        "Name of reviewer"
    ]


structured_model = model.with_structured_output(
    HospitalReview
)

st.title("AI Hospital Review Analyzer")

review = st.text_area(
    "Paste Hospital Review"
)


template = PromptTemplate(
    template="""

System Message:

You are a professional hospital review analyzer.

Extract the hospital review into structured data.

If any information is missing, return null.

Human Message:

Hospital Review:

{review}

""",

    input_variables=["review"]

)

prompt = template.invoke({

    "review": review

})


if st.button("Analyze Review"):

    if review:

        result = structured_model.invoke(prompt)

        st.subheader("Analysis Result")

        st.write("### Summary")
        st.write(result["summary"])

        st.write("### Sentiment")
        st.write(result["sentiment"])

        st.write("### Treatment Quality")
        st.write(result["treatment_quality"] or "Not Mentioned")

        st.write("### Doctor Behavior")
        st.write(result["doctor_behavior"] or "Not Mentioned")

        st.write("### Staff Service")
        st.write(result["staff_service"] or "Not Mentioned")

        st.write("### Cleanliness")
        st.write(result["cleanliness"] or "Not Mentioned")

        st.write("### Health Improvement")
        st.write(result["health_improvement"] or "Not Mentioned")

        st.write("### Recommend Hospital")
        st.write(result["recommend_hospital"])

        st.write("### Reviewer Name")
        st.write(result["reviewer_name"] or "Unknown")

    else:

        st.warning("Please enter a hospital review.")