from typing import Union, List, Optional, Dict
from adalflow.core.types import EmbedderOutput, Embedding
from FlagEmbedding import BGEM3FlagModel

class M3Embedder:
    """
    Embedder class using the BGE M3 embedding model.
    
    This class implements the interface defined in adalflow.core/embedder.py and
    returns an EmbedderOutput containing a list of Embedding objects.
    
    Args:
        model_name (str): The name or path of the BGE M3 model.
        device (str): The device to run the model (e.g., "cuda:0").
        batch_size (int): The batch size used for encoding.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda:0", batch_size: int = 64):
        super().__init__()
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name_or_path=model_name, devices=device)
        self.batch_size = batch_size

    def __call__(self, input: Union[str, List[str]], model_kwargs: Optional[Dict] = None) -> EmbedderOutput:
        """
        Embeds the provided input texts using the BGE M3 model.
        

        Args:
            input (Union[str, List[str]]): A single string or a list of strings to be encoded.
            model_kwargs (Optional[Dict]): Optional additional model parameters.
        
        Returns:
            EmbedderOutput: The output containing a list of Embedding objects,
                            model name, raw response from the model, and the original input.
        """
        # Normalize input into a list of strings.
        texts: List[str] = [input] if isinstance(input, str) else input
        
        try:
            response = self.model.encode(texts, batch_size=self.batch_size, return_dense=True)
        except Exception as e:
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=None,
                input=texts
            )
        
        # Extract embeddings from the response.
        dense_vecs = response.get("dense_vecs")
        if dense_vecs is None:
            return EmbedderOutput(
                data=[],
                error="No dense_vecs found in model response",
                raw_response=response,
                input=texts
            )

        embeddings: List[Embedding] = []
        for i, vec in enumerate(dense_vecs):
            # Convert the vector to a list of floats if needed.
            vector_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            embeddings.append(Embedding(embedding=vector_list, index=i))
        
        return EmbedderOutput(
            data=embeddings,
            model=self.model_name,
            raw_response=response,
            input=texts
        ) 