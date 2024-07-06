from modest.languages.english import English_MorphoChallenge2010, English_MorphyNet_Inflections, English_Celex


if __name__ == "__main__":
    for thing in English_Celex().generate():
        print(thing.word, "->", "/".join(thing.segment()))
    for thing in English_MorphoChallenge2010().generate():
        print(thing.word, "->", "/".join(thing.segment()))
    for thing in English_MorphyNet_Inflections().generate():
        print(thing.word, "->", "/".join(thing.segment()))
