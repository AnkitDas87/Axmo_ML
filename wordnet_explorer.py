import nltk
from nltk.corpus import wordnet as wn

# Make sure WordNet data is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def main():
    print("Welcome to the interactive WordNet explorer!")
    
    while True:
        word = input("\nEnter a word (or type 'exit' to quit): ").strip()
        if word.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Get synsets for the word
        synsets = wn.synsets(word)
        if not synsets:
            print(f"No synsets found for '{word}'. Try another word.")
            continue
        
        print(f"\nSynsets for '{word}':")
        for i, syn in enumerate(synsets, 1):
            print(f"{i}. {syn.name()} - {syn.definition()}")
        
        choice = input("\nChoose synset number to explore (or 'skip' to enter a new word): ").strip()
        if choice.lower() == 'skip':
            continue
        
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(synsets):
            print("Invalid choice. Please try again.")
            continue
        
        synset = synsets[int(choice) - 1]
        print(f"\nYou selected: {synset.name()} - {synset.definition()}")
        
        # Show examples if available
        examples = synset.examples()
        if examples:
            print("Examples:")
            for ex in examples:
                print(f" - {ex}")
        
        # Show relations
        print("\nRelations:")
        # Hypernyms
        hypernyms = synset.hypernyms()
        if hypernyms:
            print("Hypernyms:")
            for h in hypernyms:
                print(f" - {h.name()}: {h.definition()}")
        
        # Hyponyms
        hyponyms = synset.hyponyms()
        if hyponyms:
            print("Hyponyms:")
            for h in hyponyms:
                print(f" - {h.name()}: {h.definition()}")
        
        # Antonyms (only for lemmas)
        print("Antonyms:")
        antonyms_found = False
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                print(f" - {ant.name()}")
                antonyms_found = True
        if not antonyms_found:
            print(" - None found")
        
        # Similarity example with another word
        other_word = input("\nEnter another word to compute similarity (or 'skip'): ").strip()
        if other_word.lower() != 'skip':
            other_synsets = wn.synsets(other_word)
            if other_synsets:
                sim = synset.wup_similarity(other_synsets[0])
                print(f"Similarity between '{synset.name()}' and '{other_synsets[0].name()}': {sim}")
            else:
                print(f"No synsets found for '{other_word}'.")
                
if __name__ == "__main__":
    main()
