import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from torch import nn
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from lightseq.training.ops.pytorch.quantization import disable_quant
from tqdm import tqdm

#
class CoarseLM(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, logit_num, use_cond=True, cond_dim=128, cond_type='tanh'):
        super().__init__(config)
        if use_cond:
            if cond_type == 'tanh':
                self.cond_layer = nn.Sequential(
                    nn.Linear(cond_dim, config.n_embd),
                    nn.Tanh(),
                    nn.Linear(config.n_embd, config.n_embd),
                )
            else:
                self.cond_layer = nn.Sequential(
                    nn.Linear(cond_dim, config.n_embd),
                )

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, logit_num, bias=False) # quant_token_nums

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class TrainingArgs:
    def __init__(self, fp16, local_rank):
        self.fp16 = fp16
        self.local_rank = local_rank

class ModelConfig:
    def __init__(self, max_position_embeddings, hidden_size, num_attention_heads, attn_pdrop, resid_pdrop, num_hidden_layers, max_batch_tokens):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.num_hidden_layers = num_hidden_layers
        self.max_batch_tokens = max_batch_tokens

def get_hf_gpt_emb_layer_params(layer):
    init_ws = []
    init_ws.append(layer.wte.weight.detach().clone())
    init_ws.append(layer.wpe.weight.detach().clone())
    return init_ws

def inject_ls_layer(model, training_args,  config):
    from checkpoint_layer import ls_hf_gpt_enc_convert
    ls_hf_gpt_enc_convert(model, training_args, config)
    for i in range(config.num_hidden_layers):
        model.transformer.h[i].apply(disable_quant)

if __name__ == '__main__':
    device = 'cuda'

    layers = 2

    gpt2_config = GPT2Config(
        vocab_size=6*1024+1, # padding=0, text ind, codec index, 2EOS
        n_positions=6000, # 支持的最长长度
        n_ctx=6000, # 等同于n_positions
        n_embd=1024, # 隐藏层dim
        n_layer=layers, # 多少层
        n_head=16, # 多少个头
        activation_function='gelu',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-08,
        initializer_range=0.02,
        summary_type='mean',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        bos_token_id=1225,
        eos_token_id=1225,
    )

    model = CoarseLM(config=gpt2_config, logit_num=6*1024 + 1, use_cond=False).to(device)
    model.eval()

    seqs = torch.randint(low=0, high=5500, size=[1, 5500]).to(device)

    training_args = TrainingArgs(fp16=False, 
                                local_rank=0)
    config = ModelConfig(max_position_embeddings=6000, 
                         hidden_size=1024, 
                         num_attention_heads=16, 
                         attn_pdrop=0.1, 
                         resid_pdrop=0.1, 
                         num_hidden_layers=layers, 
                         max_batch_tokens = 6000)
    inject_ls_layer(model, training_args, config)

    torch.cuda.empty_cache()
    model.train()

    for i in range(10):
        lm_outputs1 = model(input_ids=seqs)
        logits1 = lm_outputs1['logits'] # [b, t, n_token]
        err1 = logits1.mean()
        err1.backward()


    for i in tqdm(range(100)):
        lm_outputs1 = model(input_ids=seqs)
        logits1 = lm_outputs1['logits'] # [b, t, n_token]
        err1 = logits1.mean()
        # err1.backward()
        print("err1:", err1)
