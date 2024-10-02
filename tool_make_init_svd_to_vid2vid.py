import sys
import os
import yaml
from safetensors.torch import load_file as load_safetensors

# assert len(sys.argv) == 2, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[1]
output_path = "ckpts/svd_xt_image_decoder_vid2vid.safetensors"

# assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *
from vtdm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

model = create_model(config_path='./configs/train-v02.yaml')

with open('./configs/train-v01.yaml', 'r') as file:
    yaml_content = yaml.safe_load(file)
    ckpt_path = yaml_content.get('model', {}).get('params', {}).get('ckpt_path', None)
    svd_ckpt = load_safetensors(ckpt_path)

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    if k in svd_ckpt:
        weights = svd_ckpt[k].clone()
        
        if 'label_emb.0.0.weight' in k:
            C1, C2 = weights.shape
            assert C2 == 768
            weights = torch.cat([
                torch.zeros_like(weights[:, :256]),
                weights[:, 512:],
            ], 1)
            
        if 'diffusion_model.input_blocks.0.0.weight' in k:
            weights_ex = [weights[:, :4]]
            for _ in range(3):
                weights_ex.append(torch.zeros_like(weights[:, :3]))
            weights_ex.append(weights[:, 4:])
            weights = torch.cat(weights_ex, 1)
            
        print(f'These weights are from svd: {k}')
    else:
        print(f'These weights are newly added: {k}')
        weights = scratch_dict[k].clone()
    target_dict[k] = weights

model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
from safetensors.torch import load_model, save_model
save_model(model, output_path)

with open('./configs/train-v02.yaml', 'r') as file:
    yaml_content = yaml.safe_load(file)

yaml_content['model']['params']['ckpt_path'] = output_path

with open('./configs/train-v02-edited.yaml', 'w') as file:
    yaml.safe_dump(yaml_content, file)
    
print('Done.')
