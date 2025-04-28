from langchain.agents import tool
import requests
from pydantic import BaseModel, Field
import datetime
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser
from langchain.document_loaders import WebBaseLoader

class OpenMeToInput(BaseModel):
    latitude: float = Field(description = "Latitude of the location to fetch weather location for ")
    longitude: float = Field(description = "Longitude of the location to fetch weather location for")
        
    

@tool(args_schema = OpenMeToInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature at given location"""
    Base_url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude" : latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1
    }
    
    response  = requests.get(Base_url, params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API request failed {response.status_code}")
        
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', "+00:00")) for time_str in results["hourly"]["time"]]
    temperature_list = results["hourly"]["temperature_2m"]
    closest_time_index = min(range(len(time_list)), key = lambda x: abs(time_list[x] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    return f"The current temperature is {current_temperature}"

from langchain.tools.render import format_tool_to_openai_function

format_tool_to_openai_function(get_current_temperature)

get_current_temperature({"latitude": 13, "longitude": 14})

import wikipedia

@tool
def search_wikipedia(query: str)-> str:
    """Run wikipedia search and get page summaries"""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles:
        try: 
            wiki_page = wikipedia.page(title = page_title, auto_suggest = False)
            summaries.append(f"page: {page_title} \n summary : {wiki_page.summary}")
        except(
            self.wiki_client_exceptions_Page_Error,
            self.wiki_client_exceptions.DisambiguationError
        ):
            pass
    if not summaries:
        return "No good wikipedia search Result was found"
    return "\n\n".join(summaries)

format_tool_to_openai_function(search_wikipedia)

text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""

from langchain.utilities.openapi import OpenAPISpec
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn

spec = OpenAPISpec.from_text(text)

pet_openai_func = openapi_spec_to_openai_fn(spec)


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish

model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0).bind(functions = pet_openai_func)

functions = [ format_tool_to_openai_function(t) for t in [search_wikipedia, get_current_temperature]]

model = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0).bind(functions = functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are helpfull but sassy assistant"),
    ("user", "{input}")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result = chain.invoke({"input": "what is the weather in vellore"})

def route(result): 
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia,
            "get_current_temperature": get_current_temperature
        }
        return tools[result.tool].run(result.tool_input)

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

chain.invoke({"input": "what is langchain"})

