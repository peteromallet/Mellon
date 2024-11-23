from huggingface_hub import list_repo_files
from transformers import T5EncoderModel, T5TokenizerFast
from utils.node_utils import NodeBase

class T5EncoderModelLoader(NodeBase):
    def execute(self, model_id, dtype, device):
        files = list_repo_files(model_id)

        # if it's an SD3.5 model, load the encoder and tokenizer from the subfolder
        if any('text_encoder_3/' in file for file in files):
            t5_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", torch_dtype=dtype)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", torch_dtype=dtype)
        else:
            t5_encoder = T5EncoderModel.from_pretrained(model_id, torch_dtype=dtype)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, torch_dtype=dtype)
        
        return { 't5_encoders': { 't5_encoder': t5_encoder, 't5_tokenizer': t5_tokenizer, 'device': device } }
