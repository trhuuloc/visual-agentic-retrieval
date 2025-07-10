import open_clip

def build_model(model_name):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    return model, preprocess