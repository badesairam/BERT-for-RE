import random
import time
import tensorflow as tf
import tensorflow_hub as hub
import bert
import os
from pathlib import Path

from bert import run_classifier
from bert import tokenization
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

label_list = [0, 1]
MAX_SEQUENCE_LENGTH = 64

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module()


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    with tf.variable_scope("loss"):
        output_layer = tf.layers.dropout(inputs=output_layer, rate=0.9)
        logits = tf.layers.dense(inputs=output_layer, units=num_labels, kernel_initializer=xavier_initializer())

        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


def model_fn_builder(num_labels, learning_rate=0.1, num_train_steps=100,
                     num_warmup_steps=20):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            logging_hook = tf.train.LoggingTensorHook({"loss":loss},every_n_iter=10)
            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  training_hooks=[logging_hook],
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def convert_single_example_RE(ex_index, example, label_list, max_seq_length,
                              tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    text_a, entity_a, entity_b = example.text_a.split('[RE]')

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    tokens_ea = tokenizer.tokenize(entity_a)
    tokens_eb = tokenizer.tokenize(entity_b)

    # Account for [CLS] and [SEP] with "- 2"
    if (len(tokens_a) + len(tokens_ea) + len(tokens_eb)) > (max_seq_length - 4) :
        tokens_a = tokens_a[0:(max_seq_length - 4 - len(tokens_ea) - len(tokens_eb))]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in tokens_ea:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_eb:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 0:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = run_classifier.InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def convert_examples_to_features_RE(examples, label_list, max_seq_length,
                                    tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`
  text is of the form sentence[RE]entity_a[RE]entity_b."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example_RE(ex_index, example, label_list,
                                            max_seq_length, tokenizer)

        features.append(feature)
    return features

def create_datasets(positive_file_path,negative_file_path):
    positive_sents = []
    negative_sents = []
    with open(positive_file_path,"r") as train_file:
        for line in train_file:
            line_split = line.split("|")
            positive_sents.append(line_split[1]+"[RE]"+line_split[5]+"[RE]"+line_split[2])

    with open(negative_file_path, "r") as test_file:
        for line in test_file:
            line_split = line.split("|")
            negative_sents.append(line_split[0] + "[RE]" + line_split[1] + "[RE]" + line_split[2])

    num_positive_sents = len(positive_sents)
    positive_labels = [1]*num_positive_sents
    num_negative_sents = len(negative_sents)
    negative_labels = [0]*num_negative_sents

    train_sents = positive_sents[:int(0.8*num_positive_sents)]+negative_sents[:int(0.8*num_negative_sents)]
    train_labels = positive_labels[:int(0.8*num_positive_sents)]+negative_labels[:int(0.8*num_negative_sents)]

    test_sents = positive_sents[int(0.8*num_positive_sents):]+negative_sents[int(0.8*num_negative_sents):]
    test_labels = positive_labels[int(0.8*num_positive_sents):]+negative_labels[int(0.8*num_negative_sents):]
    
    train_data = list(zip(train_sents,train_labels))
    test_data = list(zip(test_sents,test_labels))

    random.shuffle(train_data)
    random.shuffle(test_data)

    train_sents, train_labels = zip(*train_data)
    test_sents, test_labels = zip(*test_data)

    return train_sents,train_labels,test_sents,test_labels

train_sents,train_labels,test_sents,test_labels = create_datasets("ADE/DRUG-AE.rel","ADE/negative_data_AE.rel")

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 50
SAVE_SUMMARY_STEPS = 10
KEEP_CHECKPOINT_MAX= 5

OUTPUT_DIR ="/home/deepcompute/Bade/Relation_Extraction/model_chk_large"
#OUTPUT_DIR ="/Users/sairambade/Documents/Relation_Extraction/model_chk"

# Compute # train and warmup steps from batch size
#TODO take care of train_features
num_train_steps = int(len(train_sents) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    keep_checkpoint_max =KEEP_CHECKPOINT_MAX)

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


def train_and_evaluate(train_sents,test_sents,labels_train,labels_test):
    train_InputExamples = [ run_classifier.InputExample(guid=None,text_a=sentence,text_b=None,label=label) for sentence,label in zip(train_sents, labels_train) ]
    input_features = convert_examples_to_features_RE(train_InputExamples,label_list, MAX_SEQUENCE_LENGTH,tokenizer)
    train_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQUENCE_LENGTH, is_training=True,drop_remainder=False)
    print("############ Beginning of training #####################")
    estimator.train(input_fn=train_input_fn,max_steps=num_train_steps)
    print("############ Ending of training #####################")
    test_InputExamples = [run_classifier.InputExample(guid=None, text_a=sentence, text_b=None, label=label) for sentence, label in zip(test_sents, labels_test)]
    input_features = convert_examples_to_features_RE(test_InputExamples, label_list, MAX_SEQUENCE_LENGTH, tokenizer)
    test_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQUENCE_LENGTH, is_training=False, drop_remainder=False)
    estimator.evaluate(input_fn=test_input_fn, steps=None)



def serving_input_receiver_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

# estimator._export_to_tpu = False
# estimator.export_saved_model('saved_model',serving_input_receiver_fn)

export_dir = 'saved_model'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])

predict_fn = predictor.from_saved_model(latest)

def predict(in_sentences):
    """ predicts the output relation of sentences"""
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
    # print(input_examples)
    input_features = convert_examples_to_features_RE(input_examples, label_list, MAX_SEQUENCE_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQUENCE_LENGTH,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn, yield_single_examples=True)
    #return predictions
    return [(sentence, prediction['probabilities'], prediction['labels']) for sentence, prediction in
            zip(in_sentences, predictions)]


def predict_serve(in_sentences):
    """ predicts the output relation of sentences"""
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
    # print(input_examples)
    input_features = [convert_single_example_RE(0,input_example, label_list, MAX_SEQUENCE_LENGTH, tokenizer) for input_example in input_examples]
    input_ids = [input_feature.input_ids for input_feature in input_features]
    label_ids = [input_feature.label_id for input_feature in input_features]
    input_mask = [input_feature.input_mask for input_feature in input_features]
    segment_ids = [input_feature.segment_ids for input_feature in input_features]
    predictions = predict_fn({'input_ids':input_ids,'label_ids':label_ids,
                                'input_mask':input_mask,'segment_ids':segment_ids}) #predict_input_fn(params={"batch_size": BATCH_SIZE}))
    # return predictions
    return [(sentence,probs,label) for sentence, probs ,label in zip(in_sentences, predictions['probabilities'], predictions['labels'])]



if __name__ == "__main__":
    labels = ["Not an adverse effect", "Adverse effect"]
    matched = ["Not matched", "Matched"]
    #train_and_evaluate(train_sents,test_sents,train_labels,test_labels)
    # with open("TN_FN.txt","w") as f:
    #    for preds,actual in zip(predict(test_sents),test_labels):	
    #       if(preds[2]==0):
    #         f.write(preds[0]+"   |||   "+labels[preds[2]]+"    |||     "+labels[actual]+"    |||    "+matched[(preds[2]==actual)]+"\n")
    sents = ["treatment of philadelphia chromosome_positive acute lymphocytic leukemia with hyper_cvad and imatinib mesylate .[RE]imatinib[RE]leukemia"]
    # for i in range(32):
    #     print(predict_serve(sents))
    print(predict_serve(sents*4))
    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, ["serve"], latest)
    #     graph = tf.get_default_graph()
        # print(graph.get_operations())

    # estimator.export_saved_model('saved_model',serving_input_receiver_fn)
