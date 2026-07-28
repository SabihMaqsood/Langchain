from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


class RestaurantReview(TypedDict):

    summary: Annotated[
        str,
        "Short summary of the review"
    ]

    sentiment: Annotated[
        Literal["Positive", "Negative", "Neutral"],
        "Overall sentiment"
    ]

    food_quality: Annotated[
        Optional[str],
        "Quality of food"
    ]

    service_quality: Annotated[
        Optional[str],
        "Quality of service"
    ]

    cleanliness: Annotated[
        Optional[str],
        "Restaurant cleanliness"
    ]

    recommended_dish: Annotated[
        Optional[str],
        "Best dish mentioned in review"
    ]

    visit_again: Annotated[
        Literal["Yes", "No", "Maybe"],
        "Would customer visit again?"
    ]

    reviewer_name: Annotated[
        Optional[str],
        "Name of reviewer"
    ]


structured_model = model.with_structured_output(
    RestaurantReview
)


st.title("AI Restaurant Review Analyzer")

review = st.text_area(
    "Paste Restaurant Review"
)


template = PromptTemplate(
    template="""

System Message:

You are a professional restaurant review analyzer.

Extract the review into structured data.

If any information is missing, return null.

Human Message:

Restaurant Review:

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

        st.write("### Food Quality")
        st.write(result["food_quality"] or "Not Mentioned")

        st.write("### Service Quality")
        st.write(result["service_quality"] or "Not Mentioned")

        st.write("### Cleanliness")
        st.write(result["cleanliness"] or "Not Mentioned")

        st.write("### Recommended Dish")
        st.write(result["recommended_dish"] or "Not Mentioned")

        st.write("### Visit Again")
        st.write(result["visit_again"])

        st.write("### Reviewer Name")
        st.write(result["reviewer_name"] or "Unknown")

    else:

        st.warning("Please enter a restaurant review.")