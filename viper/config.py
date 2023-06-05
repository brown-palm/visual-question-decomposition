config = {
    "load_models": {
        "maskrcnn": False,
        "clip": False,
        "glip": True,
        "owlvit": False,
        "tcl": False,
        "gpt3_qa": True,
        "gpt3_general": True,
        "depth": True,
        "blip": True,
        "saliency": False,
        "xvlm": True,
        "codex": True,
    },
    "detect_thresholds": {"glip": 0.5, "maskrcnn": 0.8, "owlvit": 0.1},
    "ratio_box_area_to_image_area": 0.0,
    "crop_larger_margin": True,
    "verify_property": {
        "model": "xvlm",
        "thresh_clip": 0.6,
        "thresh_tcl": 0.25,
        "thresh_xvlm": 0.6,
    },
    "best_match_model": "xvlm",
    "gpt3": {
        "n_votes": 1,
        "temperature": 0.0,
        "model": "text-davinci-003",
    },
    "blip_half_precision": True,
    "blip_v2_model_type": "blip2-flan-t5-xxl",
}