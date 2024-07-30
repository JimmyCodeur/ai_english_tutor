from back.load_model import generate_phi3_response

def help_sugg(log_file_path: str, model_name: str, user_input: str) -> dict:
    print("Detected help request.")

    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        last_line = lines[-1] if lines else ""

    if last_line.startswith('[') and 'Response: ' in last_line:
        user_input = last_line.split('Response: ', 1)[1].strip()
    else:
        user_input = last_line

    prompt_for_suggestions = f"User input: {user_input}\nPlease respond with very short messages and at an easy level."

    print(prompt_for_suggestions)

    generated_suggestions = generate_phi3_response(prompt_for_suggestions)

    if isinstance(generated_suggestions, list):
        generated_suggestions = '\n'.join([f"{i + 1}. {s}" for i, s in enumerate(generated_suggestions)])

    suggestions_list = generated_suggestions.split('\n')[:5]
    print(suggestions_list)

    return {
        "user_input": user_input,
        "generated_response": None,
        "audio_base64": None,
        "suggestions": suggestions_list
    }