# fastapi_client_embedder.py
import requests # Using standard requests library
import logging
import os
from typing import Union, List, Optional, Dict, Any
from adalflow.core.types import EmbedderOutput, Embedding


logger = logging.getLogger(__name__)

class FastAPIClientEmbedder:
    """
    An Adalflow-compatible embedder that retrieves embeddings 
    from a remote FastAPI service.
    
    Args:
        base_url (str): The base URL of the FastAPI embedding service 
                        (e.g., "http://localhost:8000").
        endpoint (str): The specific API endpoint path for embedding
                        (default: "/embed/").
        request_timeout (int): Timeout in seconds for the HTTP request 
                               (default: 60).
    """
    def __init__(
        self, 
        base_url: str, 
        endpoint: str = "/embed/", 
        request_timeout: int = 60
    ):
        if not base_url:
            raise ValueError("FastAPI base_url cannot be empty.")
            
        # Ensure base_url doesn't have a trailing slash and endpoint starts with one
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint if endpoint.startswith('/') else f'/{endpoint}'
        self.full_url = f"{self.base_url}{self.endpoint}"
        self.timeout = request_timeout
        
        logger.info(f"FastAPIClientEmbedder initialized. Target URL: {self.full_url}")

    def __call__(self, input_texts: Union[str, List[str]], model_kwargs: Optional[Dict] = None) -> EmbedderOutput:
        """
        Sends text(s) to the remote FastAPI endpoint and returns embeddings.
        
        Args:
            input_texts (Union[str, List[str]]): A single string or a list of strings to embed.
            model_kwargs (Optional[Dict]): Optional dictionary.
        
        Returns:
            EmbedderOutput: An object containing the embeddings or an error message.
        """
        if model_kwargs:
             logger.warning(f"Received model_kwargs, but FastAPIClientEmbedder currently does not forward them: {model_kwargs}")

        texts: List[str] = [input_texts] if isinstance(input_texts, str) else input_texts

        if not texts:
            logger.warning("Received empty list of texts for embedding.")
            return EmbedderOutput(data=[], input=texts, error="Input text list is empty.")

        payload = {"texts": texts}
        
        response = None
        response_json = None
        
        try:
            logger.debug(f"Sending {len(texts)} text(s) to {self.full_url}")
            response = requests.post(
                self.full_url, 
                json=payload, 
                timeout=self.timeout,
                headers={"Content-Type": "application/json", "Accept": "application/json"} 
            )
            
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status() 
            
            response_json = response.json()
            logger.debug(f"Received successful response from API. Status: {response.status_code}")

            # Check for application-level errors reported in the JSON response
            api_error = response_json.get("error")
            if api_error:
                logger.error(f"FastAPI endpoint reported an error: {api_error}")
                return EmbedderOutput(
                    data=[], 
                    model=response_json.get("model"), 
                    raw_response=response_json, 
                    input=texts, 
                    error=f"API Error: {api_error}"
                )

            # Extract data and convert to Adalflow Embedding objects
            api_data = response_json.get("data", [])
            output_data: List[Embedding] = []
            if not isinstance(api_data, list):
                 logger.error(f"API response 'data' field is not a list. Found: {type(api_data)}")
                 raise ValueError("Invalid format for 'data' in API response.")
                 
            for item in api_data:
                if not isinstance(item, dict) or 'embedding' not in item or 'index' not in item:
                     logger.error(f"Invalid item format in API response data list: {item}")
                     raise ValueError("Invalid item format in 'data' list from API.")
                # Let the Embedding class handle validation/conversion of the vector
                output_data.append(Embedding(embedding=item['embedding'], index=item['index']))

            return EmbedderOutput(
                data=output_data,
                model=response_json.get("model"),
                raw_response=response_json,
                input=texts
            )

        except requests.exceptions.Timeout as e:
            logger.error(f"Request to {self.full_url} timed out after {self.timeout}s: {e}")
            return EmbedderOutput(data=[], input=texts, error=f"Request timed out: {e}", raw_response=getattr(e, 'response', None))
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to {self.full_url}: {e}")
            return EmbedderOutput(data=[], input=texts, error=f"Connection error: {e}", raw_response=getattr(e, 'response', None))
        except requests.exceptions.HTTPError as e:
            # Error raised by response.raise_for_status()
            error_detail = f"HTTP Error: {e.response.status_code} {e.response.reason}"
            # Try to get more details from the response body if possible
            try:
                error_body = e.response.json()
                error_detail += f" - Detail: {error_body.get('detail', error_body)}"
            except requests.exceptions.JSONDecodeError:
                error_detail += f" - Body: {e.response.text[:200]}" # Show beginning of text body
            logger.error(f"HTTP error received from {self.full_url}: {error_detail}")
            return EmbedderOutput(data=[], input=texts, error=error_detail, raw_response=try_get_json(e.response))
        except (requests.exceptions.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            # Errors during response parsing or data conversion
            logger.error(f"Failed to parse or process response from {self.full_url}: {e}", exc_info=True)
            raw_text = response.text if response else "No response"
            return EmbedderOutput(data=[], input=texts, error=f"Response processing error: {e}", raw_response=raw_text[:500]) # Include raw text snippet
        except Exception as e:
            # Catch any other unexpected errors
            logger.exception(f"An unexpected error occurred in FastAPIClientEmbedder: {e}")
            return EmbedderOutput(data=[], input=texts, error=f"Unexpected error: {str(e)}", raw_response=response_json if response_json else None)


def try_get_json(response: Optional[requests.Response]) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return response.text
    

if __name__ == "__main__":
    embedder = FastAPIClientEmbedder(base_url="http://localhost:8000")
    response = embedder("Hello, world!")
    print(response)

