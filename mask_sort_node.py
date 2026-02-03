"""
Mask Sort by Position - ComfyUI Custom Node
Sorts a batch of masks by their centroid X position (left to right).

This ensures face masks align with speaker order for MultiTalk.

Author: Created for Seb @ Storybook Studios
"""

import torch
import numpy as np


class MaskSortByPosition:
    """Sorts masks by X position (left to right) and optionally adds background for MultiTalk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "sort_direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {
                    "default": "left_to_right",
                    "tooltip": "Direction to sort masks.",
                }),
                "add_background": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add background mask as first channel. Try False first for MultiTalk.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("sorted_masks", "info")
    FUNCTION = "sort_masks"
    CATEGORY = "mask/utils"

    def sort_masks(self, masks, sort_direction="left_to_right", add_background=True):
        """Sort masks by centroid position and optionally add background."""
        
        # Handle different mask formats
        if masks.dim() == 2:
            # Single mask (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            # (B, C, H, W) -> (B, H, W)
            masks = masks.squeeze(1)
        
        n_masks = masks.shape[0]
        h, w = masks.shape[1], masks.shape[2]
        
        if n_masks <= 1 and not add_background:
            return (masks, f"Only {n_masks} mask(s), no sorting needed")
        
        # Calculate centroid for each mask
        centroids = []
        for i in range(n_masks):
            mask = masks[i].cpu().numpy()
            
            # Find non-zero pixels
            coords = np.where(mask > 0.5)
            
            if len(coords[0]) == 0:
                # Empty mask, use center
                cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
            else:
                # Calculate centroid
                cy = np.mean(coords[0])  # Y (row)
                cx = np.mean(coords[1])  # X (col)
            
            centroids.append((i, cx, cy))
        
        # Sort based on direction
        if sort_direction == "left_to_right":
            centroids.sort(key=lambda x: x[1])  # Sort by X ascending
        elif sort_direction == "right_to_left":
            centroids.sort(key=lambda x: -x[1])  # Sort by X descending
        elif sort_direction == "top_to_bottom":
            centroids.sort(key=lambda x: x[2])  # Sort by Y ascending
        elif sort_direction == "bottom_to_top":
            centroids.sort(key=lambda x: -x[2])  # Sort by Y descending
        
        # Reorder masks
        sorted_indices = [c[0] for c in centroids]
        sorted_masks = masks[sorted_indices]
        
        # Ensure fp32 dtype (SAM3 may output bf16 which causes autocast issues)
        sorted_masks = sorted_masks.float()
        
        # Add background mask if requested (for MultiTalk)
        # Background = inverse of all speaker masks combined
        if add_background:
            # Combine all masks
            combined = torch.zeros(h, w, device=masks.device, dtype=masks.dtype)
            for i in range(n_masks):
                combined = torch.maximum(combined, sorted_masks[i])
            
            # Background is the inverse
            background = 1.0 - combined
            background = background.unsqueeze(0)  # (1, H, W)
            
            # Prepend background to sorted masks
            # Final shape: [1 + num_speakers, H, W]
            sorted_masks = torch.cat([background, sorted_masks], dim=0)
        
        # Build info string
        info_parts = [f"Sorted {n_masks} masks ({sort_direction}):"]
        for new_idx, (old_idx, cx, cy) in enumerate(centroids):
            info_parts.append(f"  Speaker {new_idx+1}: was #{old_idx+1}, pos=({cx:.0f}, {cy:.0f})")
        
        if add_background:
            info_parts.append(f"Added background mask (total: {sorted_masks.shape[0]} channels)")
            info_parts.append(f"Shape: [{sorted_masks.shape[0]}, {h}, {w}] (MultiTalk format)")
        
        info = "\n".join(info_parts)
        print(f"[MaskSortByPosition] {info}")
        
        return (sorted_masks, info)


class MaskReorder:
    """Manually reorder masks by specifying the order."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "order": ("STRING", {
                    "default": "1,2,3",
                    "tooltip": "Comma-separated mask order. E.g., '3,1,2' puts mask 3 first.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("reordered_masks", "info")
    FUNCTION = "reorder_masks"
    CATEGORY = "mask/utils"

    def reorder_masks(self, masks, order="1,2,3"):
        """Reorder masks based on specified order."""
        
        # Handle different mask formats
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4:
            masks = masks.squeeze(1)
        
        n_masks = masks.shape[0]
        
        # Parse order string
        try:
            indices = [int(x.strip()) - 1 for x in order.split(",")]
        except ValueError:
            return (masks, f"Invalid order format: {order}")
        
        # Validate indices
        if len(indices) != n_masks:
            return (masks, f"Order has {len(indices)} values but there are {n_masks} masks")
        
        if any(i < 0 or i >= n_masks for i in indices):
            return (masks, f"Order indices out of range (1-{n_masks})")
        
        # Reorder
        reordered = masks[indices]
        
        info = f"Reordered {n_masks} masks: {order}"
        print(f"[MaskReorder] {info}")
        
        return (reordered, info)


class DtypeReset:
    """Resets PyTorch default dtype to fp32. Use after SAM3 to fix bf16 issues."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*",),  # Accept any input to control execution order
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("pass_through",)
    FUNCTION = "reset_dtype"
    CATEGORY = "utils"
    
    # This makes it work with any input type
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = False

    def reset_dtype(self, any_input):
        """Reset dtype and clear CUDA cache."""
        import torch
        import gc
        
        # Reset default dtype
        torch.set_default_dtype(torch.float32)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        print("[DtypeReset] Reset to fp32, cleared CUDA cache")
        
        return (any_input,)


class DtypeResetSimple:
    """Simple dtype reset - just triggers the reset, no passthrough needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*",),  # Optional trigger to control execution order
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "reset_dtype"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def reset_dtype(self, trigger=None):
        """Reset dtype and clear CUDA cache."""
        import torch
        import gc
        
        torch.set_default_dtype(torch.float32)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        print("[DtypeReset] Reset to fp32, cleared CUDA cache")
        
        return {}


# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MaskSortByPosition": MaskSortByPosition,
    "MaskReorder": MaskReorder,
    "DtypeReset": DtypeReset,
    "DtypeResetSimple": DtypeResetSimple,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSortByPosition": "Sort Masks by Position",
    "MaskReorder": "Reorder Masks (Manual)",
    "DtypeReset": "Dtype Reset (Pass-through)",
    "DtypeResetSimple": "Dtype Reset (Simple)",
}
