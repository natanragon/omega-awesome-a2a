# Dolphins VLM

## Overview
Vision-language model specifically designed for autonomous driving, implementing novel Grounded Chain of Thought (GCoT) reasoning for real-time scene understanding and decision making.

## Why It Matters
While many multimodal models exist, Dolphins uniquely addresses the complex requirements of autonomous driving through its innovative combination of vision-language processing and conversational abilities. Its gradient-free adaptation and reflection capabilities represent a significant step toward human-like driving intelligence.

## Technical Details
- **Base Architecture:** Enhanced OpenFlamingo
- **Inference Speed:** 1.34s on NVIDIA A100
- **Key Innovation:** Grounded Chain of Thought (GCoT) reasoning
- **Training Data:** BDD-X dataset
- **Input Processing:** Multimodal (video, text, control signals)

## Resources
- [Paper](https://arxiv.org/pdf/2312.04315.pdf)
- [Project Page](https://vlm-driver.github.io/)
- [Demo](https://vlm-driver.github.io/)

## Example Implementation
```python
# Basic inference example (pseudo-code based on paper architecture)
class DolphinsVLM:
    def __init__(self):
        self.vision_encoder = OpenFlamingo()
        self.gcot_processor = GCoTProcessor()
        
    def process_scene(self, video_frame, text_instruction, control_signals):
        # Multimodal input processing
        visual_features = self.vision_encoder(video_frame)
        
        # Apply GCoT reasoning
        reasoning_output = self.gcot_processor(
            visual_features,
            text_instruction,
            control_signals
        )
        
        return reasoning_output
