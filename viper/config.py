config = {
    "models": ["glip", "gpt3_qa", "gpt3_general", "depth", "blip", "xvlm", "codex"],
    "glip_detect_threshold": 0.5,
    "ratio_box_area_to_image_area": 0.0,
    "crop_larger_margin": True,
    "xvlm_verify_property_thresh": 0.6,
    "gpt3": {
        "n_votes": 1,
        "temperature": 0.0,
        "model": "text-davinci-003",
    },
    "blip_half_precision": True,
    "blip_v2_model_type": "blip2-flan-t5-xxl",
}
