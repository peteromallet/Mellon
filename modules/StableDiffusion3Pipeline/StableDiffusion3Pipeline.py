import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from utils.node_utils import NodeBase

class LoadSD3Transformer(NodeBase):
    def execute(self, model_id, dtype, device):
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            subfolder="transformer",
        )
 
        self.output['transformer'] = { 'model': model, 'device': device }

class LoadT5Encoder(NodeBase):
    def execute(self, model_id, device):
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
        )
        tokenizer = T5Tokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_3",
        )

        self.output['t5'] = { 'text_encoder': text_encoder, 'tokenizer': tokenizer, 'device': device }
