from nnsight import LanguageModel

if __name__ == "__main__":
    m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    with m.trace("The Eiffel Tower is in"):
        out = m.transformer.h[5].output.save()
    print("type:", type(out))
    if isinstance(out, tuple):
        print("len:", len(out), "elem types:", [type(x).__name__ for x in out])
        print("out[0] shape:", tuple(out[0].shape))
    else:
        print("shape:", tuple(out.shape))
