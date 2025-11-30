def get_caption_prompt(dataset: str) -> str:
    if dataset == "TerraIncognita":
        return ("Caption this image. I know what animal it is. Focus on other characteristics of the image, "
                "e.g., environmental characteristics, visual characteristics, camera characteristics, etc. "
                "Again, do not tell me what animal it is, give me other characteristics of the image.")
    elif dataset == "PACS":
        return ("Caption this image. I know what object it is. Focus on describing the artistic style, texture, "
                "and domain-specific details rather than the object itself. Again, do not mention the object class.")
    elif dataset == "VLCS":
        return ("Caption this image. I know what object it is. Focus on describing contextual and environmental details, "
                "such as scene composition, lighting, and background characteristics. Again, do not mention the object class.")
    elif dataset == "WILDSCamelyon":
        return ("Caption this histopathology image. I know its diagnostic category. Focus on describing tissue morphology, "
                "staining patterns, and structural details. Again, do not reveal the diagnostic category.")
    elif dataset == "WILDSFMoW":
        return ("Caption this satellite image. I know what facility type it depicts. Focus on describing environmental, geographic, "
                "and structural features present in the scene. Again, do not mention the facility type.")
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def get_difference_caption_prompt(dataset: str) -> str:
    if dataset == "TerraIncognita":
        return (
            "I am a machine learning researcher trying to figure out properties of an image beyond the image class. "
            "Give me a description of this image that is more specific than the image class. For instance, "
            "if this is an image of a bird, I don't want to know if the bird is a sparrow or a crow. I want to know "
            "if the bird is flying or sitting on a branch, if the camera is a drone or a ground-level camera, "
            "if the image is a macro shot or a close-up, or if the lighting is natural or artificial. "
            "A broader list of such properties is what I'm looking for. Give me this description for the image without focusing on the animal class."
        )
    elif dataset == "PACS":
        return (
            "I am a machine learning researcher trying to figure out properties of an image beyond the object class. "
            "Give me a description of this image that is more specific than the object class. For instance, "
            "if this is an image of a dog, I don't want to know if the dog is a bulldog or a retriever. I want to know "
            "if the scene suggests an indoor or outdoor setting, details about the artistic style, or specific texture and lighting. "
            "A broader list of such properties is what I'm looking for. Give me this description for the image without mentioning the object class."
        )
    elif dataset == "VLCS":
        return (
            "I am a researcher studying domain adaptation. Please describe this image with a focus on properties beyond the object class. "
            "For example, if this is an image of a bird, I don't want to know whether it is a sparrow or an eagle. "
            "Instead, I want detailed information about the environmental context, scene composition, lighting conditions, and background. "
            "Provide such a description without mentioning the object class."
        )
    elif dataset == "WILDSCamelyon":
        return (
            "I am a medical researcher analyzing histopathology images. Give me a description that focuses on features beyond the diagnostic category. "
            "For instance, if this image is labeled as tumor or normal, I don't want that label mentioned. Instead, describe tissue structures, "
            "staining patterns, and cellular details that reveal subtle variations in the tissue."
        )
    elif dataset == "WILDSFMoW":
        return (
            "I am a machine learning researcher working with satellite imagery. Please provide a description of this image that goes beyond the facility type. "
            "For instance, if this image depicts an airport, I don't want to simply know that it is an airport. I want to understand detailed aspects "
            "like the arrangement of structures, surrounding geographic features, patterns of urban development, and environmental conditions, "
            "without explicitly mentioning the facility type."
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")

