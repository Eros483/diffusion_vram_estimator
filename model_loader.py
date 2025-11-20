import torch
from safetensors.torch import load_file
from pathlib import Path
from transformers import CLIPTokenizer

def load_from_standard_weights(ckpt_path, device):
    """
    Load weights from HuggingFace Stable Diffusion checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint or HuggingFace model ID
        device: torch device
    
    Returns:
        Dictionary with state_dicts for encoder, decoder, diffusion, and clip
    """
    
    # Try to load as local file first
    if Path(ckpt_path).exists():
        print(f"Loading from local path: {ckpt_path}")
        if ckpt_path.endswith('.safetensors'):
            state_dict = load_file(ckpt_path, device=str(device))
        else:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        # Download from HuggingFace
        print(f"Downloading from HuggingFace: {ckpt_path}")
        from diffusers import StableDiffusionPipeline
        
        # Load the pipeline to get weights
        pipe = StableDiffusionPipeline.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        # Extract state dicts from pipeline components
        vae_state = pipe.vae.state_dict()
        unet_state = pipe.unet.state_dict()
        text_encoder_state = pipe.text_encoder.state_dict()
        
        # Convert to your format
        return {
            'encoder': convert_vae_encoder(vae_state),
            'decoder': convert_vae_decoder(vae_state),
            'diffusion': convert_unet(unet_state),
            'clip': convert_clip(text_encoder_state),
        }
    
    # If loading from merged checkpoint, split into components
    return {
        'encoder': extract_and_convert_encoder(state_dict),
        'decoder': extract_and_convert_decoder(state_dict),
        'diffusion': extract_and_convert_unet(state_dict),
        'clip': extract_and_convert_clip(state_dict),
    }

def convert_vae_encoder(vae_state):
    encoder_state = {}
    
    # 1. Handle Attention Concatenation (Mid Block)
    if 'encoder.mid_block.attentions.0.to_q.weight' in vae_state:
        q_w = vae_state['encoder.mid_block.attentions.0.to_q.weight']
        k_w = vae_state['encoder.mid_block.attentions.0.to_k.weight']
        v_w = vae_state['encoder.mid_block.attentions.0.to_v.weight']
        encoder_state['layers.13.attention.in_proj.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = vae_state['encoder.mid_block.attentions.0.to_q.bias']
        k_b = vae_state['encoder.mid_block.attentions.0.to_k.bias']
        v_b = vae_state['encoder.mid_block.attentions.0.to_v.bias']
        encoder_state['layers.13.attention.in_proj.bias'] = torch.cat([q_b, k_b, v_b], dim=0)
        
        encoder_state['layers.13.attention.out_proj.weight'] = vae_state['encoder.mid_block.attentions.0.to_out.0.weight']
        encoder_state['layers.13.attention.out_proj.bias'] = vae_state['encoder.mid_block.attentions.0.to_out.0.bias']
        encoder_state['layers.13.group_norm.weight'] = vae_state['encoder.mid_block.attentions.0.group_norm.weight']
        encoder_state['layers.13.group_norm.bias'] = vae_state['encoder.mid_block.attentions.0.group_norm.bias']

    for key, value in vae_state.items():
        if 'mid_block.attentions.0.' in key: continue
        if not key.startswith('encoder.'): continue
        
        new_key = None
        
        # Map Blocks
        if key.startswith('encoder.conv_in.'): new_key = key.replace('encoder.conv_in.', 'layers.0.')
        elif 'down_blocks.0.resnets.0.' in key: new_key = key.replace('encoder.down_blocks.0.resnets.0.', 'layers.1.')
        elif 'down_blocks.0.resnets.1.' in key: new_key = key.replace('encoder.down_blocks.0.resnets.1.', 'layers.2.')
        elif 'down_blocks.0.downsamplers.0.' in key: new_key = key.replace('encoder.down_blocks.0.downsamplers.0.conv.', 'layers.3.')
        elif 'down_blocks.1.resnets.0.' in key: new_key = key.replace('encoder.down_blocks.1.resnets.0.', 'layers.4.')
        elif 'down_blocks.1.resnets.1.' in key: new_key = key.replace('encoder.down_blocks.1.resnets.1.', 'layers.5.')
        elif 'down_blocks.1.downsamplers.0.' in key: new_key = key.replace('encoder.down_blocks.1.downsamplers.0.conv.', 'layers.6.')
        elif 'down_blocks.2.resnets.0.' in key: new_key = key.replace('encoder.down_blocks.2.resnets.0.', 'layers.7.')
        elif 'down_blocks.2.resnets.1.' in key: new_key = key.replace('encoder.down_blocks.2.resnets.1.', 'layers.8.')
        elif 'down_blocks.2.downsamplers.0.' in key: new_key = key.replace('encoder.down_blocks.2.downsamplers.0.conv.', 'layers.9.')
        elif 'down_blocks.3.resnets.0.' in key: new_key = key.replace('encoder.down_blocks.3.resnets.0.', 'layers.10.')
        elif 'down_blocks.3.resnets.1.' in key: new_key = key.replace('encoder.down_blocks.3.resnets.1.', 'layers.11.')
        elif 'mid_block.resnets.0.' in key: new_key = key.replace('encoder.mid_block.resnets.0.', 'layers.12.')
        elif 'mid_block.resnets.1.' in key: new_key = key.replace('encoder.mid_block.resnets.1.', 'layers.14.')
        
        # FIX: Handle both norm_out and conv_norm_out
        elif 'norm_out' in key:
            # This catches encoder.norm_out AND encoder.conv_norm_out
            new_key = 'layers.15.' + key.split('.')[-1]
            
        elif key.startswith('encoder.conv_out.'): 
            new_key = key.replace('encoder.conv_out.', 'layers.17.')
        
        if new_key:
            new_key = new_key.replace('norm1.', 'group_norm_1.')
            new_key = new_key.replace('norm2.', 'group_norm_2.')
            new_key = new_key.replace('conv1.', 'conv_1.')
            new_key = new_key.replace('conv2.', 'conv_2.')
            new_key = new_key.replace('conv_shortcut.', 'residual_layer.')
            encoder_state[new_key] = value
            
    # FIX: Check for encoder.quant_conv (not just quant_conv)
    if 'encoder.quant_conv.weight' in vae_state:
        encoder_state['layers.18.weight'] = vae_state['encoder.quant_conv.weight']
        encoder_state['layers.18.bias'] = vae_state['encoder.quant_conv.bias']
    elif 'quant_conv.weight' in vae_state: # Fallback
        encoder_state['layers.18.weight'] = vae_state['quant_conv.weight']
        encoder_state['layers.18.bias'] = vae_state['quant_conv.bias']

    return encoder_state

def convert_vae_decoder(vae_state):
    decoder_state = {}
    
    # 1. Handle Attention Concatenation
    if 'decoder.mid_block.attentions.0.to_q.weight' in vae_state:
        q_w = vae_state['decoder.mid_block.attentions.0.to_q.weight']
        k_w = vae_state['decoder.mid_block.attentions.0.to_k.weight']
        v_w = vae_state['decoder.mid_block.attentions.0.to_v.weight']
        decoder_state['layers.3.attention.in_proj.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = vae_state['decoder.mid_block.attentions.0.to_q.bias']
        k_b = vae_state['decoder.mid_block.attentions.0.to_k.bias']
        v_b = vae_state['decoder.mid_block.attentions.0.to_v.bias']
        decoder_state['layers.3.attention.in_proj.bias'] = torch.cat([q_b, k_b, v_b], dim=0)
        
        decoder_state['layers.3.attention.out_proj.weight'] = vae_state['decoder.mid_block.attentions.0.to_out.0.weight']
        decoder_state['layers.3.attention.out_proj.bias'] = vae_state['decoder.mid_block.attentions.0.to_out.0.bias']
        decoder_state['layers.3.group_norm.weight'] = vae_state['decoder.mid_block.attentions.0.group_norm.weight']
        decoder_state['layers.3.group_norm.bias'] = vae_state['decoder.mid_block.attentions.0.group_norm.bias']

    for key, value in vae_state.items():
        if 'mid_block.attentions.0.' in key: continue
        if not key.startswith('decoder.'): continue
        
        new_key = None
        if key.startswith('decoder.conv_in.'): new_key = key.replace('decoder.conv_in.', 'layers.1.')
        elif 'mid_block.resnets.0.' in key: new_key = key.replace('decoder.mid_block.resnets.0.', 'layers.2.')
        elif 'mid_block.resnets.1.' in key: new_key = key.replace('decoder.mid_block.resnets.1.', 'layers.4.')
        elif 'up_blocks.0.resnets.0.' in key: new_key = key.replace('decoder.up_blocks.0.resnets.0.', 'layers.5.')
        elif 'up_blocks.0.resnets.1.' in key: new_key = key.replace('decoder.up_blocks.0.resnets.1.', 'layers.6.')
        elif 'up_blocks.0.resnets.2.' in key: new_key = key.replace('decoder.up_blocks.0.resnets.2.', 'layers.7.')
        elif 'up_blocks.0.upsamplers.0.' in key: new_key = key.replace('decoder.up_blocks.0.upsamplers.0.conv.', 'layers.8.conv.')
        elif 'up_blocks.1.resnets.0.' in key: new_key = key.replace('decoder.up_blocks.1.resnets.0.', 'layers.9.')
        elif 'up_blocks.1.resnets.1.' in key: new_key = key.replace('decoder.up_blocks.1.resnets.1.', 'layers.10.')
        elif 'up_blocks.1.resnets.2.' in key: new_key = key.replace('decoder.up_blocks.1.resnets.2.', 'layers.11.')
        elif 'up_blocks.1.upsamplers.0.' in key: new_key = key.replace('decoder.up_blocks.1.upsamplers.0.conv.', 'layers.12.conv.')
        elif 'up_blocks.2.resnets.0.' in key: new_key = key.replace('decoder.up_blocks.2.resnets.0.', 'layers.13.')
        elif 'up_blocks.2.resnets.1.' in key: new_key = key.replace('decoder.up_blocks.2.resnets.1.', 'layers.14.')
        elif 'up_blocks.2.resnets.2.' in key: new_key = key.replace('decoder.up_blocks.2.resnets.2.', 'layers.15.')
        elif 'up_blocks.2.upsamplers.0.' in key: new_key = key.replace('decoder.up_blocks.2.upsamplers.0.conv.', 'layers.16.conv.')
        elif 'up_blocks.3.resnets.0.' in key: new_key = key.replace('decoder.up_blocks.3.resnets.0.', 'layers.17.')
        elif 'up_blocks.3.resnets.1.' in key: new_key = key.replace('decoder.up_blocks.3.resnets.1.', 'layers.18.')
        elif 'up_blocks.3.resnets.2.' in key: new_key = key.replace('decoder.up_blocks.3.resnets.2.', 'layers.19.')
        
        # FIX: Handle both norm_out and conv_norm_out
        elif 'norm_out' in key:
             new_key = 'layers.20.' + key.split('.')[-1]
             
        elif key.startswith('decoder.conv_out.'): new_key = key.replace('decoder.conv_out.', 'layers.22.')

        if new_key:
            new_key = new_key.replace('norm1.', 'group_norm_1.')
            new_key = new_key.replace('norm2.', 'group_norm_2.')
            new_key = new_key.replace('conv1.', 'conv_1.')
            new_key = new_key.replace('conv2.', 'conv_2.')
            new_key = new_key.replace('conv_shortcut.', 'residual_layer.')
            decoder_state[new_key] = value
            
    if 'decoder.post_quant_conv.weight' in vae_state:
        decoder_state['layers.0.weight'] = vae_state['decoder.post_quant_conv.weight']
        decoder_state['layers.0.bias'] = vae_state['decoder.post_quant_conv.bias']
    elif 'post_quant_conv.weight' in vae_state: # Fallback
        decoder_state['layers.0.weight'] = vae_state['post_quant_conv.weight']
        decoder_state['layers.0.bias'] = vae_state['post_quant_conv.bias']

    return decoder_state

def convert_unet(unet_state):
    """Convert UNet weights to match your Diffusion model architecture"""
    diffusion_state = {}
    
    for key, value in unet_state.items():
        new_key = key
        
        # --- 1. PREFIX MAPPING (Locating the block) ---
        
        # Time embedding
        if key.startswith('time_embedding.'):
            new_key = 'time_embedding.' + key.split('time_embedding.')[1]
        
        # Conv in
        elif key.startswith('conv_in.'):
            new_key = 'unet.encoders.0.0.' + key.split('conv_in.')[1]
        
        # Down blocks -> encoders
        elif key.startswith('down_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            
            if block_idx == 0:
                if 'resnets.0.' in key: new_key = key.replace('down_blocks.0.resnets.0.', 'unet.encoders.1.0.')
                elif 'attentions.0.' in key: new_key = key.replace('down_blocks.0.attentions.0.', 'unet.encoders.1.1.')
                elif 'resnets.1.' in key: new_key = key.replace('down_blocks.0.resnets.1.', 'unet.encoders.2.0.')
                elif 'attentions.1.' in key: new_key = key.replace('down_blocks.0.attentions.1.', 'unet.encoders.2.1.')
                elif 'downsamplers.0.' in key: new_key = key.replace('down_blocks.0.downsamplers.0.conv.', 'unet.encoders.3.0.')
            elif block_idx == 1:
                if 'resnets.0.' in key: new_key = key.replace('down_blocks.1.resnets.0.', 'unet.encoders.4.0.')
                elif 'attentions.0.' in key: new_key = key.replace('down_blocks.1.attentions.0.', 'unet.encoders.4.1.')
                elif 'resnets.1.' in key: new_key = key.replace('down_blocks.1.resnets.1.', 'unet.encoders.5.0.')
                elif 'attentions.1.' in key: new_key = key.replace('down_blocks.1.attentions.1.', 'unet.encoders.5.1.')
                elif 'downsamplers.0.' in key: new_key = key.replace('down_blocks.1.downsamplers.0.conv.', 'unet.encoders.6.0.')
            elif block_idx == 2:
                if 'resnets.0.' in key: new_key = key.replace('down_blocks.2.resnets.0.', 'unet.encoders.7.0.')
                elif 'attentions.0.' in key: new_key = key.replace('down_blocks.2.attentions.0.', 'unet.encoders.7.1.')
                elif 'resnets.1.' in key: new_key = key.replace('down_blocks.2.resnets.1.', 'unet.encoders.8.0.')
                elif 'attentions.1.' in key: new_key = key.replace('down_blocks.2.attentions.1.', 'unet.encoders.8.1.')
                elif 'downsamplers.0.' in key: new_key = key.replace('down_blocks.2.downsamplers.0.conv.', 'unet.encoders.9.0.')
            elif block_idx == 3:
                if 'resnets.0.' in key: new_key = key.replace('down_blocks.3.resnets.0.', 'unet.encoders.10.0.')
                elif 'resnets.1.' in key: new_key = key.replace('down_blocks.3.resnets.1.', 'unet.encoders.11.0.')
        
        # Mid block -> bottleneck
        elif key.startswith('mid_block.'):
            if 'resnets.0.' in key: new_key = key.replace('mid_block.resnets.0.', 'unet.bottleneck.0.')
            elif 'attentions.0.' in key: new_key = key.replace('mid_block.attentions.0.', 'unet.bottleneck.1.')
            elif 'resnets.1.' in key: new_key = key.replace('mid_block.resnets.1.', 'unet.bottleneck.2.')
        
        # Up blocks -> decoders
        elif key.startswith('up_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            
            if block_idx == 0:
                if 'resnets.0.' in key: new_key = key.replace('up_blocks.0.resnets.0.', 'unet.decoders.0.0.')
                elif 'resnets.1.' in key: new_key = key.replace('up_blocks.0.resnets.1.', 'unet.decoders.1.0.')
                elif 'resnets.2.' in key: new_key = key.replace('up_blocks.0.resnets.2.', 'unet.decoders.2.0.')
                elif 'upsamplers.0.' in key: new_key = key.replace('up_blocks.0.upsamplers.0.conv.', 'unet.decoders.2.1.conv.')
            elif block_idx == 1:
                if 'resnets.0.' in key: new_key = key.replace('up_blocks.1.resnets.0.', 'unet.decoders.3.0.')
                elif 'attentions.0.' in key: new_key = key.replace('up_blocks.1.attentions.0.', 'unet.decoders.3.1.')
                elif 'resnets.1.' in key: new_key = key.replace('up_blocks.1.resnets.1.', 'unet.decoders.4.0.')
                elif 'attentions.1.' in key: new_key = key.replace('up_blocks.1.attentions.1.', 'unet.decoders.4.1.')
                elif 'resnets.2.' in key: new_key = key.replace('up_blocks.1.resnets.2.', 'unet.decoders.5.0.')
                elif 'attentions.2.' in key: new_key = key.replace('up_blocks.1.attentions.2.', 'unet.decoders.5.1.')
                elif 'upsamplers.0.' in key: new_key = key.replace('up_blocks.1.upsamplers.0.conv.', 'unet.decoders.5.2.conv.')
            elif block_idx == 2:
                if 'resnets.0.' in key: new_key = key.replace('up_blocks.2.resnets.0.', 'unet.decoders.6.0.')
                elif 'attentions.0.' in key: new_key = key.replace('up_blocks.2.attentions.0.', 'unet.decoders.6.1.')
                elif 'resnets.1.' in key: new_key = key.replace('up_blocks.2.resnets.1.', 'unet.decoders.7.0.')
                elif 'attentions.1.' in key: new_key = key.replace('up_blocks.2.attentions.1.', 'unet.decoders.7.1.')
                elif 'resnets.2.' in key: new_key = key.replace('up_blocks.2.resnets.2.', 'unet.decoders.8.0.')
                elif 'attentions.2.' in key: new_key = key.replace('up_blocks.2.attentions.2.', 'unet.decoders.8.1.')
                elif 'upsamplers.0.' in key: new_key = key.replace('up_blocks.2.upsamplers.0.conv.', 'unet.decoders.8.2.conv.')
            elif block_idx == 3:
                if 'resnets.0.' in key: new_key = key.replace('up_blocks.3.resnets.0.', 'unet.decoders.9.0.')
                elif 'attentions.0.' in key: new_key = key.replace('up_blocks.3.attentions.0.', 'unet.decoders.9.1.')
                elif 'resnets.1.' in key: new_key = key.replace('up_blocks.3.resnets.1.', 'unet.decoders.10.0.')
                elif 'attentions.1.' in key: new_key = key.replace('up_blocks.3.attentions.1.', 'unet.decoders.10.1.')
                elif 'resnets.2.' in key: new_key = key.replace('up_blocks.3.resnets.2.', 'unet.decoders.11.0.')
                elif 'attentions.2.' in key: new_key = key.replace('up_blocks.3.attentions.2.', 'unet.decoders.11.1.')
        
        # Final Output
        elif key.startswith('conv_norm_out.'):
            new_key = key.replace('conv_norm_out.', 'final.group_norm.')
        elif key.startswith('conv_out.'):
            new_key = key.replace('conv_out.', 'final.conv.')

        # --- 2. SUFFIX MAPPING (Inside the block) ---

        # A. Handle RESNET Blocks
        if 'resnets' in key:
            new_key = new_key.replace('norm1.', 'group_norm_feature.')
            new_key = new_key.replace('norm2.', 'group_norm_merged.')
            new_key = new_key.replace('conv1.', 'conv_feature.')
            new_key = new_key.replace('conv2.', 'conv_merged.')
            new_key = new_key.replace('time_emb_proj.', 'linear_time.')
            new_key = new_key.replace('conv_shortcut.', 'residual_layer.')

        # B. Handle ATTENTION Blocks
        elif 'attentions' in key:
            # Strip the nested transformer container
            new_key = new_key.replace('transformer_blocks.0.', '')
            
            # --- NEW FIXES HERE ---
            # Map Outer Attention Container layers
            new_key = new_key.replace('proj_in.', 'conv_input.')
            new_key = new_key.replace('proj_out.', 'conv_output.')
            new_key = new_key.replace('.norm.', '.group_norm.') 
            # ----------------------

            # Handle Self Attention (ATTN1) Merge (Q, K, V)
            if 'attn1.to_q' in key:
                k_key = key.replace('to_q', 'to_k')
                v_key = key.replace('to_q', 'to_v')
                combined = torch.cat([value, unet_state[k_key], unet_state[v_key]], dim=0)
                
                new_key = new_key.replace('attn1.to_q.', 'attention_1.in_proj.')
                new_key = new_key.replace('norm1.', 'layer_norm_1.') 
                diffusion_state[new_key] = combined
                continue
            
            if 'attn1.to_k' in key or 'attn1.to_v' in key:
                continue
                
            # Standard Attention Mappings (Transformer Internals)
            new_key = new_key.replace('norm1.', 'layer_norm_1.')
            new_key = new_key.replace('norm2.', 'layer_norm_2.')
            new_key = new_key.replace('norm3.', 'layer_norm_3.')
            
            new_key = new_key.replace('attn1.to_out.0.', 'attention_1.out_proj.')
            
            new_key = new_key.replace('attn2.to_q.', 'attention_2.q_proj.')
            new_key = new_key.replace('attn2.to_k.', 'attention_2.k_proj.')
            new_key = new_key.replace('attn2.to_v.', 'attention_2.v_proj.')
            new_key = new_key.replace('attn2.to_out.0.', 'attention_2.o_proj.')
            
            new_key = new_key.replace('ff.net.0.proj.', 'linear_geglu_1.')
            new_key = new_key.replace('ff.net.2.', 'linear_geglu_2.')

        # C. Handle Global/Final mappings
        else:
            new_key = new_key.replace('norm1.', 'layer_norm_1.')
            new_key = new_key.replace('norm2.', 'layer_norm_2.')

        diffusion_state[new_key] = value
    
    return diffusion_state

def convert_clip(text_encoder_state):
    clip_state = {}
    
    for key, value in text_encoder_state.items():
        new_key = key
        new_key = new_key.replace('text_model.', '')
        
        # FIX: Parameter (no .weight) vs Embedding (has .weight)
        if 'embeddings.position_embedding.weight' in key:
            new_key = new_key.replace('embeddings.position_embedding.weight', 'embedding.position_embedding')
            clip_state[new_key] = value
            continue

        if 'self_attn.q_proj' in key:
            k_key = key.replace('q_proj', 'k_proj')
            v_key = key.replace('q_proj', 'v_proj')
            combined = torch.cat([value, text_encoder_state[k_key], text_encoder_state[v_key]], dim=0)
            new_key = new_key.replace('encoder.layers.', 'layers.')
            new_key = new_key.replace('self_attn.q_proj.', 'attention.in_proj.')
            new_key = new_key.replace('layer_norm1.', 'layer_norm_1.')
            clip_state[new_key] = combined
            continue
            
        if 'self_attn.k_proj' in key or 'self_attn.v_proj' in key:
            continue
            
        new_key = new_key.replace('embeddings.token_embedding', 'embedding.token_embedding')
        new_key = new_key.replace('encoder.layers.', 'layers.')
        new_key = new_key.replace('self_attn.out_proj.', 'attention.out_proj.')
        new_key = new_key.replace('layer_norm1.', 'layer_norm_1.')
        new_key = new_key.replace('layer_norm2.', 'layer_norm_2.')
        new_key = new_key.replace('mlp.fc1.', 'linear_1.')
        new_key = new_key.replace('mlp.fc2.', 'linear_2.')
        new_key = new_key.replace('final_layer_norm', 'layernorm')
        
        clip_state[new_key] = value
    
    return clip_state


def extract_and_convert_encoder(state_dict):
    """Extract encoder from merged checkpoint"""
    encoder_dict = {}
    for key, value in state_dict.items():
        if 'first_stage_model.encoder' in key:
            new_key = key.replace('first_stage_model.encoder.', '')
            encoder_dict[new_key] = value
    return convert_vae_encoder({'encoder.' + k: v for k, v in encoder_dict.items()})


def extract_and_convert_decoder(state_dict):
    """Extract decoder from merged checkpoint"""
    decoder_dict = {}
    for key, value in state_dict.items():
        if 'first_stage_model.decoder' in key:
            new_key = key.replace('first_stage_model.decoder.', '')
            decoder_dict[new_key] = value
    return convert_vae_decoder({'decoder.' + k: v for k, v in decoder_dict.items()})


def extract_and_convert_unet(state_dict):
    """Extract UNet from merged checkpoint"""
    unet_dict = {}
    for key, value in state_dict.items():
        if 'model.diffusion_model' in key:
            new_key = key.replace('model.diffusion_model.', '')
            unet_dict[new_key] = value
    return convert_unet(unet_dict)


def extract_and_convert_clip(state_dict):
    """Extract CLIP from merged checkpoint"""
    clip_dict = {}
    for key, value in state_dict.items():
        if 'cond_stage_model.transformer' in key:
            new_key = key.replace('cond_stage_model.transformer.', '')
            clip_dict[new_key] = value
    return convert_clip(clip_dict)


def preload_models_from_standard_weights(ckpt_path, device):
    """
    Convenience function to load all models with weights.
    
    Usage:
        models = preload_models_from_standard_weights(
            "runwayml/stable-diffusion-v1-5", 
            device="cuda"
        )
    """
    # Import your model classes - make sure these match your actual module names
    # If all models are in the same file, adjust imports accordingly
    try:
        from encoder import VAE_Encoder
        from decoder import VAE_Decoder
        from diffusion import Diffusion
        from clip import CLIP
    except ImportError:
        # If models are in the current namespace (same notebook)
        import __main__
        VAE_Encoder = __main__.VAE_Encoder
        VAE_Decoder = __main__.VAE_Decoder
        Diffusion = __main__.Diffusion
        CLIP = __main__.CLIP
    
    print("Loading state dicts from checkpoint...")
    state_dicts = load_from_standard_weights(ckpt_path, device)
    
    print("Initializing models...")
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)
    clip = CLIP().to(device)
    
    print("Loading weights into models...")
    # Use strict=False to see which keys don't match
    missing_e, unexpected_e = encoder.load_state_dict(state_dicts['encoder'], strict=False)
    missing_d, unexpected_d = decoder.load_state_dict(state_dicts['decoder'], strict=False)
    missing_diff, unexpected_diff = diffusion.load_state_dict(state_dicts['diffusion'], strict=False)
    missing_c, unexpected_c = clip.load_state_dict(state_dicts['clip'], strict=False)
    
    if missing_e or unexpected_e:
        print(f"Encoder - Missing: {len(missing_e)}, Unexpected: {len(unexpected_e)}")
        if len(missing_e) < 20:
            print(f"  Missing keys: {missing_e}")
        if len(unexpected_e) < 20:
            print(f"  Unexpected keys: {unexpected_e}")
    else:
        print("✓ Encoder loaded successfully")
        
    if missing_d or unexpected_d:
        print(f"Decoder - Missing: {len(missing_d)}, Unexpected: {len(unexpected_d)}")
        if len(missing_d) < 20:
            print(f"  Missing keys: {missing_d}")
        if len(unexpected_d) < 20:
            print(f"  Unexpected keys: {unexpected_d}")
    else:
        print("✓ Decoder loaded successfully")
        
    if missing_diff or unexpected_diff:
        print(f"Diffusion - Missing: {len(missing_diff)}, Unexpected: {len(unexpected_diff)}")
        if len(missing_diff) < 20:
            print(f"  Missing keys: {missing_diff}")
        if len(unexpected_diff) < 20:
            print(f"  Unexpected keys: {unexpected_diff}")
    else:
        print("✓ Diffusion loaded successfully")
        
    if missing_c or unexpected_c:
        print(f"CLIP - Missing: {len(missing_c)}, Unexpected: {len(unexpected_c)}")
        if len(missing_c) < 20:
            print(f"  Missing keys: {missing_c}")
        if len(unexpected_c) < 20:
            print(f"  Unexpected keys: {unexpected_c}")
    else:
        print("✓ CLIP loaded successfully")
    
    print("\nAll models loaded!")
    
    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


def load_tokenizer():
    """Load CLIP tokenizer from HuggingFace"""
    return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")