import numpy as np
import jax.numpy as jnp
from jax_mae import mae_vit_base_patch16 as mae
from flax.training import train_state
from flax.core.frozen_dict import freeze, unfreeze
from flax.serialization import msgpack_restore
from collections import defaultdict
import optax
import jax
import pdb

def print_keys_recursively(obj, prefix=""):
    if isinstance(obj, dict):
        for key in obj:
            print_keys_recursively(obj[key], prefix + f"{key}.")
    else:
        print(f"{prefix[:-1]}: {obj.shape}")

def nested_dict():
    return defaultdict(nested_dict)


def modify_based_on_key_string(key_string):
    # Modify parameters based on key
    
    if key_string.startswith('vit.encoder.layer.'):
        change_in_layer = {
            'attention.attention.query.kernel':'Attention_0.Dense_0.kernel',
            'attention.attention.query.bias':'Attention_0.Dense_0.bias',
            'attention.attention.key.kernel':'Attention_0.Dense_1.kernel',
            'attention.attention.key.bias':'Attention_0.Dense_1.bias',
            'attention.attention.value.kernel':'Attention_0.Dense_2.kernel',
            'attention.attention.value.bias':'Attention_0.Dense_2.bias',
            'attention.output.dense.kernel':'Attention_0.Dense_3.kernel',
            'attention.output.dense.bias':'Attention_0.Dense_3.bias',
            'intermediate.dense.kernel':'Mlp_0.Dense_0.kernel',
            'intermediate.dense.bias':'Mlp_0.Dense_0.bias',
            'output.dense.kernel':'Mlp_0.Dense_1.kernel',
            'output.dense.bias':'Mlp_0.Dense_1.bias',
            'layernorm_before.scale':'LayerNorm_0.scale',
            'layernorm_before.bias':'LayerNorm_0.bias',
            'layernorm_after.scale':'LayerNorm_1.scale',
            'layernorm_after.bias':'LayerNorm_1.bias'
            }
        enc_id = int(key_string.split('.')[3])
        enc_layer = ".".join(key_string.split('.')[4:])
        return 'encoder_block_%02d.%s'%(int(enc_id), change_in_layer[enc_layer])
    elif key_string.startswith('vision_model.encoder.layers.'):
        change_in_layer = {
            'self_attn.q_proj.kernel':'Attention_0.Dense_0.kernel',
            'self_attn.q_proj.bias':'Attention_0.Dense_0.bias',
            'self_attn.k_proj.kernel':'Attention_0.Dense_1.kernel',
            'self_attn.k_proj.bias':'Attention_0.Dense_1.bias',
            'self_attn.v_proj.kernel':'Attention_0.Dense_2.kernel',
            'self_attn.v_proj.bias':'Attention_0.Dense_2.bias',
            'self_attn.out_proj.kernel':'Attention_0.Dense_3.kernel',
            'self_attn.out_proj.bias':'Attention_0.Dense_3.bias',
            'mlp.fc1.kernel':'Mlp_0.Dense_0.kernel',
            'mlp.fc1.bias':'Mlp_0.Dense_0.bias',
            'mlp.fc2.kernel':'Mlp_0.Dense_1.kernel',
            'mlp.fc2.bias':'Mlp_0.Dense_1.bias',
            'layer_norm1.scale':'LayerNorm_0.scale',
            'layer_norm1.bias':'LayerNorm_0.bias',
            'layer_norm2.scale':'LayerNorm_1.scale',
            'layer_norm2.bias':'LayerNorm_1.bias'
            }
        enc_id = int(key_string.split('.')[3])
        enc_layer = ".".join(key_string.split('.')[4:])
        return 'encoder_block_%02d.%s'%(int(enc_id), change_in_layer[enc_layer])
    else:
        return None


def build_nested_dict(keys, value, current_dict):
    if len(keys) == 1:
        current_dict[keys[0]] = value
    else:
        build_nested_dict(keys[1:], value, current_dict[keys[0]])


def copy_and_modify_nested_dict(original_dict, new_dict, prefix=""):
    for key, value in original_dict.items():
        new_key_string = prefix + key
        
        if isinstance(value, dict):
            copy_and_modify_nested_dict(value, new_dict, new_key_string + '.')
        else:
            new_key_string = modify_based_on_key_string(new_key_string)
            if new_key_string is not None:
                keys = new_key_string.split('.')
                build_nested_dict(keys, value, new_dict)




def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def load_jax_vit_base_model():
    # Load pretrained ViT model
    from transformers import FlaxCLIPModel
    clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    module = clip.module
    variables = {'params': clip.params}
    vision_model, vision_model_vars = module.bind(variables).vision_model.unbind()
    state_dict = {'vision_model': vision_model_vars['params']}
    pretrain_params = nested_dict()
    copy_and_modify_nested_dict(state_dict, pretrain_params)
    pretrain_params = defaultdict_to_dict(pretrain_params)
    return pretrain_params

# if __name__ == '__main__':
#     with open("/home/ustc/Research/huggingface.co/clip-vit-base-patch16/flax_model.msgpack", 'rb') as f:
#         state_dict = msgpack_restore(f.read())
#     # print_keys_recursively(state_dict)
#     # pdb.set_trace()

#     rng = jax.random.PRNGKey(0)
#     rng, init_rng = jax.random.split(rng)
#     state, variables = create_train_state(init_rng)

#     pretrain_params = nested_dict()
#     copy_and_modify_nested_dict(state_dict, pretrain_params)
#     # print_keys_recursively(pretrain_params)
#     variables = unfreeze(variables)
    
#     N = {"on": 0, "off": 0}
#     for key, value in pretrain_params.items():
#         if variables['params'].keys():
#             variables['params'][key] = value
#             N["on"] += 1
#         else:
#             N["off"] += 1
#     print(N)
#     resume_params = variables['params']
#     state = state.replace(params=freeze(resume_params))