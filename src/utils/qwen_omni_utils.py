"""
Utility functions for Qwen2.5-Omni multimodal processing.
"""

def process_mm_info(messages, use_audio_in_video=False):
    """
    Process multimodal information from messages.
    
    Args:
        messages: List of message dictionaries with content
        use_audio_in_video: Boolean flag for audio in video processing
    
    Returns:
        Tuple of (audios, images, videos) lists
    """
    audios = []
    images = []
    videos = []
    
    for message in messages:
        if "content" in message:
            for content_item in message["content"]:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "audio" and "audio" in content_item:
                        audios.append(content_item["audio"])
                    elif content_item.get("type") == "image" and "image" in content_item:
                        images.append(content_item["image"])
                    elif content_item.get("type") == "video" and "video" in content_item:
                        videos.append(content_item["video"])
    
    return audios, images, videos