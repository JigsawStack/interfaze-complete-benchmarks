# main.py
from src.commons import invoke_interfaze
from pydantic import BaseModel, Field


class ResponseModel(BaseModel):
    capital: str = Field(..., description="The capital of the country", required=True)


response = invoke_interfaze(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    structured_response=True,
    structure_definition=ResponseModel,
)
print(response.choices[0].message.content)
