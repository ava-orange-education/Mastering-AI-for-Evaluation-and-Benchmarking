from deepeval import evaluate from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric from sklearn.metrics
import classification_report

———————————————————

DeepEval: LLM Evaluation

———————————————————

Create an evaluation test case

test_case = LLMTestCase( input=“What is RAG?”, actual_output=“RAG
combines information retrieval with language generation.” )

Define the evaluation metric

metric = AnswerRelevancyMetric( threshold=0.7 )

Run the DeepEval evaluation

evaluate( test_cases=[test_case], metrics=[metric] )

———————————————————

Scikit-learn: Class-Level Evaluation

———————————————————

Example actual and predicted class labels.

Replace these lists with labels from your classification model.

actual = [ “Network”, “Database”, “Application”, “Authentication”,
“Infrastructure”, “Security”]

predicted = [ “Network”, “Database”, “Application”, “Authentication”,
“Infrastructure”, “Security”]

Generate the classification report

print( classification_report( actual, predicted, target_names=[
“Network”, “Database”, “Application”, “Authentication”,
“Infrastructure”, “Security” ] ) )
