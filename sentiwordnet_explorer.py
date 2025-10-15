import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

nltk.download('sentiwordnet')
nltk.download('wordnet')

def main():
    print("Welcome to the improved interactive SentiWordNet explorer!")

    while True:
        word = input("\nEnter a word to analyze sentiment (or type 'exit' to quit): ").strip()
        if word.lower() == 'exit':
            print("Goodbye!")
            break

        synsets = wn.synsets(word)
        if not synsets:
            print(f"No synsets found for '{word}'. Try another word.")
            continue

        print(f"\nSynsets for '{word}':")
        for i, syn in enumerate(synsets, 1):
            print(f"{i}. {syn.name()} - {syn.definition()}")

        choice = input("\nChoose synset number to analyze sentiment (or 'skip' to enter a new word): ").strip()
        if choice.lower() == 'skip':
            continue

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(synsets):
            print("Invalid choice. Please try again.")
            continue

        synset = synsets[int(choice) - 1]
        print(f"\nYou selected: {synset.name()} - {synset.definition()}")

        # Use senti_synsets(word, pos) and match synset by definition to find correct sentiment
        pos = synset.pos()
        senti_synsets = list(swn.senti_synsets(word, pos))

        # Find matching senti_synset with the same synset name
        matched = None
        for senti_syn in senti_synsets:
            if senti_syn.synset.name() == synset.name():
                matched = senti_syn
                break

        if matched:
            print(f"\nSentiment Scores for '{synset.name()}':")
            print(f"Positive score: {matched.pos_score()}")
            print(f"Negative score: {matched.neg_score()}")
            print(f"Objective score: {matched.obj_score()}")
        else:
            print("No sentiment data found for this synset.")

        analyze_another = input("\nAnalyze another word? (yes/no): ").strip().lower()
        if analyze_another not in ('yes', 'y'):
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
