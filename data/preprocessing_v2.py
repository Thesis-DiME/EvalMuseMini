import json
import pandas as pd


def filter_unique_prompts(input_file, output_file):
    """
    Reads a metadata.json file, filters unique rows based on the 'prompt' field,
    and writes the result to a new JSON file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    try:
        # Step 1: Read the input JSON file
        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Step 2: Filter unique rows based on the 'prompt' field
        seen_prompts = set()
        unique_data = []

        for entry in data:
            prompt = entry.get("prompt")  # Extract the 'prompt' field
            if prompt and prompt not in seen_prompts:
                seen_prompts.add(prompt)
                unique_data.append(entry)

        # Step 3: Write the filtered data to the output JSON file
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(unique_data, file, indent=4, ensure_ascii=False)

        print(f"Filtered data saved to {output_file}")

    except json.JSONDecodeError:
        print("Error: The input file is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    import os
    cwd = os.getcwd()
    print(cwd)
    input_file = "data/dataset/mini_test.json"
    output_file = "data/dataset/prompts.json"
    
    input_file = os.path.join(cwd, input_file)
    filter_unique_prompts(input_file, output_file)


if __name__ == "__main__":
    main()
