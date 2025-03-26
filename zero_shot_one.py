from transformers import pipeline

pipe = pipeline(model="facebook/bart-large-mnli")
pipe(
    "I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
