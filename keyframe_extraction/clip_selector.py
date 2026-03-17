"""CLIP-based keyframe selection with batched embeddings for efficiency."""
import numpy as np
import torch
import clip
from PIL import Image
import cv2
from typing import List, Optional


class CLIPKeyframeSelector:
    """Select keyframes using CLIP semantic embeddings (optimized with batching)."""
    
    def __init__(self, model_name: str = "ViT-B/32", device=None, batch_size: int = 32):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.batch_size = batch_size
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"[OK] CLIP model loaded: {model_name} (batch_size={batch_size})")
    
    def extract_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Extract embedding for a single frame (legacy method, use batch version for multiple frames)."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def extract_embeddings_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple frames in batches (much faster than per-frame).
        
        Args:
            frames: List of frames as numpy arrays (BGR or grayscale)
            
        Returns:
            Array of shape (N, D) where N is number of frames and D is embedding dimension
        """
        if len(frames) == 0:
            return np.array([])
        
        all_embeddings = []
        batch_tensors: List[torch.Tensor] = []

        def _flush_batch() -> None:
            if not batch_tensors:
                return
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            batch_tensors.clear()
            with torch.inference_mode():
                embeddings = self.model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())

        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            img_tensor = self.preprocess(pil_img)
            batch_tensors.append(img_tensor)
            if len(batch_tensors) >= self.batch_size:
                _flush_batch()

        _flush_batch()

        if not all_embeddings:
            return np.array([])
        return torch.cat(all_embeddings, dim=0).numpy()
    
    def select_keyframe(self, frames: List[np.ndarray], quality_scores: List[float],
                       quality_weight: float = 0.3) -> int:
        """
        Select keyframe using CLIP embeddings and quality scores.
        
        Uses batched embedding extraction for efficiency.
        """
        if len(frames) <= 1:
            return 0
        
        # Use batched embedding extraction (much faster)
        embeddings = self.extract_embeddings_batch(frames)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # Compute distances to centroid
        distances = np.array([np.linalg.norm(e - centroid) for e in embeddings])
        
        # Normalize distance scores
        if distances.max() > distances.min():
            dist_scores = 1.0 - (distances - distances.min()) / (distances.max() - distances.min())
        else:
            dist_scores = np.ones(len(distances))
        
        # Normalize quality scores
        quality_array = np.array(quality_scores)
        if quality_array.max() > quality_array.min():
            qual_scores = (quality_array - quality_array.min()) / (quality_array.max() - quality_array.min())
        else:
            qual_scores = np.ones(len(quality_scores))
        
        # Combine scores
        combined = (1 - quality_weight) * dist_scores + quality_weight * qual_scores
        return int(np.argmax(combined))
