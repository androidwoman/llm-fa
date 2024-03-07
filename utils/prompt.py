def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''پاسخ این سوال چیست؟
            ### سوال:
            {instruction}
            ### ورودی:
            {input_text}
            ### پاسخ:
            {response}
            '''
        else:
            text = f'''پاسخ این سوال چیست؟
            ### سوال:
            {instruction}
            ### پاسخ:
            {response}
            '''
        output_text.append(text)

    return output_text