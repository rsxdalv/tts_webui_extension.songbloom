# Copyright (c) Cypress Yang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modified for TTS WebUI extension with Gradio interface.

import os
import torch
import torchaudio
import gradio as gr
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

# Disable flash attention for compatibility
os.environ['DISABLE_FLASH_ATTN'] = "1"

NAME2REPO = {
    "songbloom_full_150s": "CypressYang/SongBloom",
    "songbloom_full_150s_dpo": "CypressYang/SongBloom"
}

class SongBloomInterface:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
        self.current_model_name = None
        self.sample_rate = 44100
        
    def hf_download(self, model_name: str = "songbloom_full_150s", **kwargs) -> Dict[str, str]:
        """Download model files from Hugging Face Hub."""
        repo_id = NAME2REPO[model_name]
        
        try:
            cfg_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{model_name}.yaml", 
                local_dir=str(self.cache_dir), 
                **kwargs
            )
            ckpt_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{model_name}.pt", 
                local_dir=str(self.cache_dir), 
                **kwargs
            )
            
            vae_cfg_path = hf_hub_download(
                repo_id="CypressYang/SongBloom", 
                filename="stable_audio_1920_vae.json", 
                local_dir=str(self.cache_dir), 
                **kwargs
            )
            vae_ckpt_path = hf_hub_download(
                repo_id="CypressYang/SongBloom", 
                filename="autoencoder_music_dsp1920.ckpt", 
                local_dir=str(self.cache_dir), 
                **kwargs
            )
            
            g2p_path = hf_hub_download(
                repo_id="CypressYang/SongBloom", 
                filename="vocab_g2p.yaml", 
                local_dir=str(self.cache_dir), 
                **kwargs
            )
            
            return {
                "config": cfg_path,
                "checkpoint": ckpt_path,
                "vae_config": vae_cfg_path,
                "vae_checkpoint": vae_ckpt_path,
                "g2p": g2p_path
            }
        except Exception as e:
            raise ValueError(f"Failed to download model {model_name}: {str(e)}")

    def load_config(self, cfg_file: str) -> DictConfig:
        """Load configuration file with proper resolvers."""
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
        OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
        OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", str(self.cache_dir)))
        
        file_cfg = OmegaConf.load(open(cfg_file, 'r')) if cfg_file is not None else OmegaConf.create()
        return file_cfg

    def load_model(self, model_name: str, dtype: str = "float32") -> str:
        """Load the SongBloom model."""
        
        from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler

        # Download model if not already loaded
        if self.current_model_name != model_name:
            self.hf_download(model_name)
            
            cfg_path = self.cache_dir / f"{model_name}.yaml"
            cfg = self.load_config(str(cfg_path))
            cfg.max_dur = cfg.max_dur + 20
            
            dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
            self.model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype_torch)
            self.model.set_generation_params(**cfg.inference)
            self.sample_rate = self.model.sample_rate
            self.current_model_name = model_name
            
        return f"✅ Model {model_name} loaded successfully!"
            

    def generate_music(
        self,
        lyrics: str,
        prompt_audio: Optional[str],
        model_name: str,
        dtype: str,
        progress=gr.Progress()
    ) -> Tuple[str, Optional[str]]:
        """Generate a single music sample from lyrics and prompt audio."""
        
        if not lyrics.strip():
            gr.Info("Please provide lyrics.")
            return None
        
        if prompt_audio is None:
            gr.Info("Please upload a prompt audio file.")
            return None
        
        try:
            # Load model
            progress(0.1, desc="Loading model...")
            load_status = self.load_model(model_name, dtype)
            if load_status.startswith("❌"):
                gr.Info(load_status)
                return None

            if self.model is None:
                gr.Info("Model failed to load.")
                return None

            progress(0.3, desc="Processing audio...")
            
            # Load and process prompt audio
            prompt_wav, sr = torchaudio.load(prompt_audio)
            if sr != self.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.sample_rate)
            
            # Convert to mono and limit duration to 10 seconds
            dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype_torch)
            prompt_wav = prompt_wav[..., :10 * self.sample_rate]
            
            progress(0.5, desc="Generating music...")
            
            # Generate single sample
            wav = self.model.generate(lyrics, prompt_wav)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f"_sample_0.flac", delete=False) as tmp_file:
                output_path = tmp_file.name
                torchaudio.save(output_path, wav[0].cpu().float(), self.sample_rate)

            progress(1.0, desc="Complete!")
            
            status_msg = f"✅ Generated 1 sample successfully!"
            return status_msg, output_path
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            return error_msg, None

# Global interface instance
songbloom_interface = SongBloomInterface()


def songbloom_ui():
    """Create the SongBloom UI - compatible with the existing main.py structure."""
    
    gr.Markdown("""
    # 🎵 SongBloom Music Generation (Current version needs 40GB VRAM)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=list(NAME2REPO.keys()),
                value="songbloom_full_150s",
                label="Model",
                info="Choose the SongBloom model variant"
            )
            
            # Lyrics input
            lyrics_input = gr.Textbox(
                label="Lyrics",
                placeholder="Enter your lyrics here...",
                lines=6,
                info="The lyrics for the song you want to generate"
            )
            
            # Audio prompt upload
            prompt_audio = gr.Audio(
                label="Prompt Audio",
                type="filepath",
            )
            
            with gr.Row():
                dtype_radio = gr.Radio(
                    choices=["float32", "bfloat16"],
                    value="float32",
                    label="Precision",
                    info="Model precision (bfloat16 uses less memory)"
                )
                
                # Removed samples slider: single-sample generation only
                
                generate_btn = gr.Button(
                    "🎵 Generate Music",
                    variant="primary",
                    size="lg",
                )
                
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            
            audio_output = gr.Audio(
                label="Sample 1",
                interactive=False
            )
    
    generate_btn.click(
        fn=songbloom_interface.generate_music,
        inputs=[
            lyrics_input,
            prompt_audio,
            model_dropdown,
            dtype_radio,
        ],
        outputs=[audio_output],
    )
    
if __name__ == "__main__":
    # Standalone launch
    with gr.Blocks(title="SongBloom Music Generation") as demo:
        songbloom_ui()
    
    demo.launch(
        server_port=7772,
    )
