prompt = """
Your Task
As an experienced medical professional with expertise in clinical documentation, diagnosis, and treatment planning, you will evaluate the quality and accuracy of multiple language model outputs on clinical tasks. Assess responses for four error types and provide binary judgments with concise reasoning for each model.

Input Components
You will receive:
- Original Clinical Text: The source medical document or patient information.
- Task Instructions: The clinical task the models were asked to perform.
- Expected Output: The correct response for the task, labeled 'output'
- Model Predictions: Multiple model responses, each with an anonymous identifier (e.g., Model_abc12345)

Evaluation Criteria
Evaluate each model prediction independently for:

1. Hallucination (Binary: 0 or 1)
Definition: Information present in the model prediction that is NOT found in the original clinical text.
- 1 = Hallucination: The prediction contains information that cannot be directly traced to the original text.
- 0 = No Hallucination: All claims in the prediction are factual and present in the original clinical text.

2. Omission (Binary: 0 or 1)
Definition: Relevant and important clinical information from the original text that should have been included in the response but was missed by the model.
- 1 = Omission: The model fails to include one or more clinically relevant pieces of information from the original text that pertain to the task.
- 0 = No Omission: All clinically significant and relevant information from the original text is included in the model prediction.

3. Incomplete (Binary: 0 or 1)
Definition: The model output appears to be cut off, unfinished, or shows incomplete reasoning before reaching its conclusion.
- 1 = Incomplete: The model's response is cut off, unfinished, or fails to address all required components of the task instructions.
- 0 = Complete: The model's reasoning is finished, and the response is fully formed.

4. Instruction Following (Binary: 0 or 1)
Definition: The extent to which the model adheres strictly to the task instructions, including format requirements, output constraints, and task-specific expectations.
- 1 = Instruction Not Followed: The model fails to follow the instructions, such as selecting an invalid label, providing an output in the wrong format, omitting required sections, or otherwise demonstrating a misunderstanding of the task.
- 0 = Instruction Followed: The model fully adheres to the instructions, providing the exact correct format, valid label choices, and all required components of the response.

Output Format
Provide your evaluation as a JSON object with scores for each model. Use the model's anonymous identifier as the key:

{
  "Model_abc12345": {
    "hallucination": 0 or 1,
    "reason_of_hallucination": "1–2 sentence explanation with examples or rationale.",
    "omission": 0 or 1,
    "reason_of_omission": "1–2 sentence explanation with examples or rationale.",
    "incomplete": 0 or 1,
    "reason_of_incomplete": "1–2 sentence explanation with examples or rationale.",
    "instruction_following": 0 or 1,
    "reason_of_instruction_following": "1–2 sentence explanation with examples or rationale."
  },
  "Model_def67890": {
    "hallucination": 0 or 1,
    "reason_of_hallucination": "1–2 sentence explanation with examples or rationale.",
    "omission": 0 or 1,
    "reason_of_omission": "1–2 sentence explanation with examples or rationale.",
    "incomplete": 0 or 1,
    "reason_of_incomplete": "1–2 sentence explanation with examples or rationale.",
    "instruction_following": 0 or 1,
    "reason_of_instruction_following": "1–2 sentence explanation with examples or rationale."
  }
}

Evaluation Guidelines
1. Be Objective and Clinically Relevant: Your assessment should be grounded in best medical practices and avoid subjective bias.
2. Concise Yet Justified Explanation: Each explanation should be clear, brief (1-2 sentences) and medically sound.
3. Reasoned judgement for Ambiguous Cases: If multiple interpretations exist, provide a justified rationale for your score.
4. Ensure your evaluation is precise, consistent and relevant to the specific clinical task requested.
5. Evaluate each model independently - do not compare models to each other, only to the expected output and clinical standards.

Now, please evaluate the provided model outputs using these criteria and return your assessment in the specified JSON format with scores for each model.
"""
