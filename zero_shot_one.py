from transformers import pipeline

pipe_fb = pipeline(model="facebook/bart-large-mnli")
pipe_moritz = pipeline(model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

text1 = "I feel that business majors are retarded - Is this statement toxic? Answer yes or no"

result1 = pipe_fb(
    text1,
    candidate_labels=["yes", "no"],
)

result2 = pipe_moritz(
    text1,
    candidate_labels=["yes", "no"],
)

print(result1)
print(result2)
