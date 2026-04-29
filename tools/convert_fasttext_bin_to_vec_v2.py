import fasttext
import sys
import time
import os

def convert_bin_to_vec(input_path, output_path):
    print(f"Loading {input_path} with fasttext...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return
        
    try:
        model = fasttext.load_model(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    words = model.get_words()
    dim = model.get_dimension()
    n_words = len(words)
    print(f"Loaded model. Vocabulary size: {n_words}")
    
    start_time = time.time()
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"{n_words} {dim}\n")
        
        for i, word in enumerate(words):
            if i % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i+1) / elapsed if elapsed > 0 else 0
                print(f"Processed {i}/{n_words} words... ({rate:.0f} w/s)     ", end='\r')
            
            vector = model.get_word_vector(word)
            # Faster string formatting
            vector_str = " ".join(map(str, vector))
            f_out.write(f"{word} {vector_str}\n")
            
    print(f"\nSuccessfully converted to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_fasttext_bin_to_vec.py <input.bin> <output.vec>")
    else:
        convert_bin_to_vec(sys.argv[1], sys.argv[2])
