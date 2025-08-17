from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal,Optional
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation", huggingfacehub_api_token=huggingface_api)

model=ChatHuggingFace(llm=llm)

# schema

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg","N"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


review="""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Heavy and uncomfortable for one-handed use
Bloatware from Samsung’s One UI
High price point at $1,300
"""

structured_model=model.with_structured_output(Review)

response=structured_model.invoke(review)
if response is None:
    print("Parser failed, falling back to raw model output...")
    raw = model.invoke(
        f"Summarize the text and give sentiment. Output as JSON {{'summary': ..., 'sentiment': ...}}:\n{review}"
    )
    print(raw.content)
else:
    print(response)