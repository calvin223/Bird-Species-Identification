import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
import cv2
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention maps from ResNet-50 CBAM model"""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: ResNet50CBAM model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Store hooks
        self.attention_maps = {}
        self.feature_maps = {}
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks to capture attention maps"""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'cbam'):
                    # Store input feature map
                    self.feature_maps[name] = input[0].detach().cpu()
                    
                    # Compute and store attention maps
                    with torch.no_grad():
                        channel_att = module.cbam.channel_attention(input[0])
                        spatial_att = module.cbam.spatial_attention(input[0] * channel_att)
                        
                        self.attention_maps[name] = {
                            'channel': channel_att.detach().cpu(),
                            'spatial': spatial_att.detach().cpu(),
                            'combined': (channel_att * spatial_att).detach().cpu()
                        }
            return hook
        
        # Register hooks on all blocks with CBAM
        for name, module in self.model.named_modules():
            if hasattr(module, 'cbam') and ('layer' in name):
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_maps(self, spectrogram: torch.Tensor) -> Dict:
        """
        Get attention maps for a given spectrogram
        
        Args:
            spectrogram: Input spectrogram tensor (1, 1, 128, 259)
        
        Returns:
            Dictionary containing attention maps for each layer
        """
        self.attention_maps.clear()
        self.feature_maps.clear()
        
        # Register hooks
        self.register_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(spectrogram.to(self.device))
        
        # Remove hooks
        self.remove_hooks()
        
        return self.attention_maps
    
    def visualize_attention_progression(self, 
                                      spectrogram: torch.Tensor,
                                      save_path: Optional[str] = None,
                                      layer_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Visualize attention maps progression through layers
        
        Args:
            spectrogram: Input spectrogram tensor
            save_path: Path to save the figure
            layer_names: Specific layers to visualize (default: all)
        
        Returns:
            Matplotlib figure
        """
        attention_maps = self.get_attention_maps(spectrogram)
        
        if layer_names is None:
            layer_names = sorted(attention_maps.keys())
        
        n_layers = len(layer_names)
        fig, axes = plt.subplots(3, n_layers, figsize=(4*n_layers, 12))
        
        if n_layers == 1:
            axes = axes.reshape(-1, 1)
        
        # Original spectrogram
        spec_np = spectrogram.squeeze().cpu().numpy()
        
        for idx, layer_name in enumerate(layer_names):
            if layer_name not in attention_maps:
                continue
            
            att_maps = attention_maps[layer_name]
            
            # Channel attention (visualize as 1D)
            channel_att = att_maps['channel'].squeeze().numpy()
            if len(channel_att.shape) > 1:
                channel_att = channel_att.mean(axis=(1, 2))
            
            axes[0, idx].bar(range(len(channel_att)), channel_att)
            axes[0, idx].set_title(f'{layer_name}\nChannel Attention')
            axes[0, idx].set_xlabel('Channel')
            axes[0, idx].set_ylabel('Attention Weight')
            
            # Spatial attention
            spatial_att = att_maps['spatial'].squeeze().numpy()
            
            # Resize spatial attention to match spectrogram size
            spatial_att_resized = cv2.resize(spatial_att, (spec_np.shape[1], spec_np.shape[0]))
            
            im1 = axes[1, idx].imshow(spatial_att_resized, cmap='hot', aspect='auto')
            axes[1, idx].set_title('Spatial Attention')
            axes[1, idx].set_xlabel('Time')
            axes[1, idx].set_ylabel('Frequency')
            plt.colorbar(im1, ax=axes[1, idx])
            
            # Spectrogram with attention overlay
            axes[2, idx].imshow(spec_np, cmap='viridis', aspect='auto', alpha=0.7)
            im2 = axes[2, idx].imshow(spatial_att_resized, cmap='hot', aspect='auto', alpha=0.3)
            axes[2, idx].set_title('Spectrogram + Attention')
            axes[2, idx].set_xlabel('Time')
            axes[2, idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_frequency_attention(self, 
                                    spectrogram: torch.Tensor,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize frequency-wise attention if model uses it
        
        Args:
            spectrogram: Input spectrogram tensor
            save_path: Path to save the figure
        
        Returns:
            Matplotlib figure
        """
        if not hasattr(self.model, 'freq_attention'):
            print("Model does not have frequency attention module")
            return None
        
        # Hook to capture frequency attention
        freq_attention = None
        
        def freq_hook(module, input, output):
            nonlocal freq_attention
            freq_attention = output.detach().cpu()
        
        # Register hook
        hook = self.model.freq_attention.register_forward_hook(freq_hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(spectrogram.to(self.device))
        
        # Remove hook
        hook.remove()
        
        if freq_attention is None:
            return None
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Original spectrogram
        spec_np = spectrogram.squeeze().cpu().numpy()
        ax1.imshow(spec_np, cmap='viridis', aspect='auto')
        ax1.set_title('Original Spectrogram')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        
        # Frequency attention pattern
        freq_att = freq_attention.squeeze().mean(dim=0).numpy()  # Average over channels
        if len(freq_att.shape) > 1:
            freq_att = freq_att.mean(axis=1)  # Average over time if needed
        
        ax2.plot(freq_att, range(len(freq_att)))
        ax2.set_xlabel('Attention Weight')
        ax2.set_ylabel('Frequency Bin')
        ax2.set_title('Frequency-wise Attention')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Match spectrogram orientation
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_attention_heatmap(self,
                               spectrogram: torch.Tensor,
                               layer_name: str,
                               attention_type: str = 'spatial') -> np.ndarray:
        """
        Create a heatmap for specific attention
        
        Args:
            spectrogram: Input spectrogram
            layer_name: Name of the layer
            attention_type: 'spatial', 'channel', or 'combined'
        
        Returns:
            Attention heatmap as numpy array
        """
        attention_maps = self.get_attention_maps(spectrogram)
        
        if layer_name not in attention_maps:
            raise ValueError(f"Layer {layer_name} not found in attention maps")
        
        att_map = attention_maps[layer_name][attention_type].squeeze().numpy()
        
        # Resize to match spectrogram dimensions
        spec_shape = spectrogram.squeeze().shape
        if attention_type == 'spatial' and att_map.shape != spec_shape:
            att_map = cv2.resize(att_map, (spec_shape[1], spec_shape[0]))
        
        return att_map
    
    def save_attention_summary(self,
                             spectrogram: torch.Tensor,
                             species_name: str,
                             output_dir: str):
        """
        Save a comprehensive attention summary for a sample
        
        Args:
            spectrogram: Input spectrogram
            species_name: Name of the bird species
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all attention maps
        attention_maps = self.get_attention_maps(spectrogram)
        
        # Create summary figure
        fig = plt.figure(figsize=(20, 12))
        
        # Original spectrogram
        plt.subplot(3, 4, 1)
        spec_np = spectrogram.squeeze().cpu().numpy()
        plt.imshow(spec_np, cmap='viridis', aspect='auto')
        plt.title(f'Spectrogram: {species_name}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()
        
        # Select key layers to visualize
        key_layers = []
        for name in attention_maps.keys():
            if 'layer1' in name and len([k for k in key_layers if 'layer1' in k]) == 0:
                key_layers.append(name)
            elif 'layer2' in name and len([k for k in key_layers if 'layer2' in k]) == 0:
                key_layers.append(name)
            elif 'layer3' in name and len([k for k in key_layers if 'layer3' in k]) == 0:
                key_layers.append(name)
            elif 'layer4' in name and len([k for k in key_layers if 'layer4' in k]) == 0:
                key_layers.append(name)
        
        # Visualize spatial attention for key layers
        for idx, layer_name in enumerate(key_layers[:6]):
            plt.subplot(3, 4, idx + 2)
            spatial_att = attention_maps[layer_name]['spatial'].squeeze().numpy()
            spatial_att_resized = cv2.resize(spatial_att, (spec_np.shape[1], spec_np.shape[0]))
            plt.imshow(spatial_att_resized, cmap='hot', aspect='auto')
            plt.title(f'{layer_name} Spatial')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
        
        # Average attention across all layers
        plt.subplot(3, 4, 8)
        all_spatial_atts = []
        for att_maps in attention_maps.values():
            spatial_att = att_maps['spatial'].squeeze().numpy()
            spatial_att_resized = cv2.resize(spatial_att, (spec_np.shape[1], spec_np.shape[0]))
            all_spatial_atts.append(spatial_att_resized)
        
        avg_attention = np.mean(all_spatial_atts, axis=0)
        plt.imshow(avg_attention, cmap='hot', aspect='auto')
        plt.title('Average Spatial Attention')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()
        
        # Attention-weighted spectrogram
        plt.subplot(3, 4, 9)
        weighted_spec = spec_np * avg_attention
        plt.imshow(weighted_spec, cmap='viridis', aspect='auto')
        plt.title('Attention-weighted Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()
        
        # Frequency attention profile
        plt.subplot(3, 4, 10)
        freq_profile = avg_attention.mean(axis=1)
        plt.plot(freq_profile, range(len(freq_profile)))
        plt.xlabel('Average Attention')
        plt.ylabel('Frequency Bin')
        plt.title('Frequency Attention Profile')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        
        # Time attention profile
        plt.subplot(3, 4, 11)
        time_profile = avg_attention.mean(axis=0)
        plt.plot(time_profile)
        plt.xlabel('Time Step')
        plt.ylabel('Average Attention')
        plt.title('Time Attention Profile')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{species_name}_attention_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save attention maps as numpy arrays for further analysis
        np.save(output_dir / f'{species_name}_avg_attention.npy', avg_attention)
        
        print(f"Saved attention summary for {species_name} to {output_dir}")