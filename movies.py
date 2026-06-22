from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from config import API_KEY

MODEL_NAME= "openrouter:deepseek/deepseek-v3.2"
model = init_chat_model(
    MODEL_NAME,
    api_key=API_KEY,
)

class Actor(BaseModel):
    """An actor with details."""
    name: str = Field(description="The name of the actor")
    birth_year: int = Field(description="The year the actor was born")

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")
    synopsis: str = Field(description="A brief summary of the movie's plot")
    stars: list[Actor] = Field(description="A list of the main actors in the movie")

movie_title = input("Enter a movie title: ")

structured_model = model.with_structured_output(Movie, method="json_schema")
response = structured_model.invoke(f"Provide details about the movie {movie_title}.")

print(response)