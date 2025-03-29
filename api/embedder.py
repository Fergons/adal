# embedder.py
# (Paste the second M3Embedder class implementation here)
# Make sure BGEM3FlagModel is importable or defined as well
# For standalone example, let's redefine the necessary types if adalflow isn't installed

from typing import Union, List, Optional, Dict, Any
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import logging
from adalflow.core.types import EmbedderOutput, Embedding

logger = logging.getLogger(__name__)


class M3Embedder:
    """
    Embedder class using the BGE M3 embedding model.
    
    This class implements the interface defined in adalflow.core/embedder.py and
    returns an EmbedderOutput containing a list of Embedding objects.
    
    Args:
        model_name (str): The name or path of the BGE M3 model.
        device (str): The device to run the model (e.g., "cuda:0", "cpu").
        batch_size (int): The batch size used for encoding.
        max_length (int): The maximum sequence length.
        normalize_embeddings (bool): Whether to normalize dense embeddings.
    """
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3", 
                 device: str = "cuda:0", 
                 batch_size: int = 64, 
                 max_length: int = 8192, # BGE M3 supports up to 8192
                 normalize_embeddings: bool = True):
        super().__init__()
        self.model_name = model_name
        self.device = device # Store device
        self.batch_size = batch_size
        self.max_length = max_length
        logger.info(f"Initializing BGEM3FlagModel '{model_name}' on device '{device}'...")
        # Pass relevant parameters to BGEM3FlagModel constructor
        # Check BGEM3FlagModel's __init__ signature for available parameters
        self.model = BGEM3FlagModel(
             model_name_or_path=model_name, 
             pooling_method='cls', # or other pooling methods if needed
             normalize_embeddings=normalize_embeddings, # Pass normalization flag
             use_fp16=True if 'cuda' in device else False, # Use FP16 on GPU by default
             device=device # Directly specify the device here if supported, else handle manually
             # query_max_length=max_length, # Check if these specific length args exist
             # passage_max_length=max_length, # or if a single 'max_length' is used internally
        )
        # If BGEM3FlagModel doesn't take 'device' directly, you might need to handle it here:
        # if device:
        #    self.model.model.to(device) # Move the underlying torch model
        #    logger.info(f"Model moved to device: {device}")

        logger.info("BGEM3FlagModel initialized successfully.")


    def __call__(self, input_texts: Union[str, List[str]], model_kwargs: Optional[Dict] = None) -> EmbedderOutput:
        """
        Embeds the provided input texts using the BGE M3 model.
        
        Args:
            input_texts (Union[str, List[str]]): A single string or a list of strings to be encoded.
            model_kwargs (Optional[Dict]): Optional additional model parameters 
                                           passed to the underlying encode method.
        
        Returns:
            EmbedderOutput: The output containing a list of Embedding objects,
                            model name, raw response from the model, and the original input.
        """
        # Normalize input into a list of strings.
        texts: List[str] = [input_texts] if isinstance(input_texts, str) else input_texts
        
        # Merge default/instance kwargs with call-specific kwargs if any
        encode_kwargs = {
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "return_dense": True,
            "return_sparse": False, # Assuming we only need dense for this API
            "return_colbert_vecs": False 
        }
        if model_kwargs:
            encode_kwargs.update(model_kwargs)

        try:
            # Ensure device context if needed, although BGEM3FlagModel might handle it
            # with torch.cuda.device(self.device): # Example if manual device context is needed
            response = self.model.encode(texts, **encode_kwargs)

        except Exception as e:
            logger.error(f"Error during BGE M3 encoding: {e}", exc_info=True)
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=None,
                input=texts,
                model=self.model_name
            )
        
        # Extract embeddings from the response.
        dense_vecs = response.get("dense_vecs")
        if dense_vecs is None:
            logger.error("No 'dense_vecs' found in BGE M3 model response.")
            return EmbedderOutput(
                data=[],
                error="No dense_vecs found in model response",
                raw_response=response,
                input=texts,
                model=self.model_name
            )

        embeddings: List[Embedding] = []
        if isinstance(dense_vecs, np.ndarray):
             for i, vec in enumerate(dense_vecs):
                 # Convert numpy array row to list of floats
                 vector_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                 embeddings.append(Embedding(embedding=vector_list, index=i))
        else:
             # Handle cases where it might not be a numpy array (though it usually is)
             logger.warning(f"Received dense_vecs of unexpected type: {type(dense_vecs)}")
             # Attempt conversion if possible, otherwise return error
             try:
                for i, vec in enumerate(dense_vecs):
                    vector_list = list(map(float, vec)) # Attempt basic conversion
                    embeddings.append(Embedding(embedding=vector_list, index=i))
             except Exception as conv_e:
                 logger.error(f"Could not convert dense_vecs elements to float lists: {conv_e}")
                 return EmbedderOutput(
                    data=[],
                    error=f"Could not process dense_vecs of type {type(dense_vecs)}",
                    raw_response=response,
                    input=texts,
                    model=self.model_name
                )

        
        return EmbedderOutput(
            data=embeddings,
            model=self.model_name,
            raw_response=response, # Keep raw if needed, else set to None
            input=texts
        )