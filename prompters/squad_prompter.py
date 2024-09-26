class SQuAD_Prompter(object):
    def __init__(self):
        self.instruction = "Instruction: Using the provided context, answer the question with exact phrases and avoid explanations."
        self.question_sep = "\nQuestion: "
        self.context_sep = "\nContext: "
        self.answer_sep = "\nAnswer: "
        self.template = "Context: {context}\n---\nQuestion: {question}\n---\nAnswer: "
        self.template = self.instruction + "\n---\n" + self.template

    def generate_prompt(self, question, context, answer=None):
        prompt = self.template.format(question=question, context=context)
        if answer:
            prompt += answer
        return prompt

    def get_response(self, text, eos_token=None):
        response = text.split(self.answer_sep)[1]
        if eos_token:
            response = response.split(eos_token)[0]
        return response.strip()
