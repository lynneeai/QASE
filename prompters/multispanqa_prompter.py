class MultiSpanQA_Prompter(object):
    def __init__(self, instruction=None):
        if instruction:
            self.instruction = f"Instruction: {instruction}"
        else:
            self.instruction = "Instruction: Using the provided context, answer the question with exact phrases and avoid explanations."
        self.context_sep = "\nContext: "
        self.question_sep = "\nQuestion: "
        self.answer_sep = "\nAnswer: "
        self.template = "Context: {context}\n---\nQuestion: {question}\n---\nAnswer: "
        self.template = self.instruction + "\n---\n" + self.template

    def generate_prompt(self, question, context, spans=None):
        prompt = self.template.format(question=question, context=context)
        if spans:
            prompt += spans
        return prompt

    def get_context(self, text):
        context = text.split("\n---" + self.question_sep)[0]
        return context.split(self.context_sep)[1]

    def get_question(self, text):
        question = text.split("\n---" + self.answer_sep)[0]
        return question.split(self.question_sep)[1]

    def get_response(self, text, eos_token=None):
        response = text.split(self.answer_sep)[1]
        if eos_token:
            response = response.split(eos_token)[0]
        return response.strip()
