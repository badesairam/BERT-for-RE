from __future__ import absolute_import, division, print_function

import argparse
import math
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
											  BertConfig,BertPreTrainedModel,BertModel)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.metrics import classification_report,f1_score,accuracy_score,precision_score,recall_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

MAX_SEQ_LENGTH = 128
label_list = ["Adverse effect","Not an adverse effect"]
num_labels = len(label_list)+1


class BertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config, num_labels):
		super(BertForSequenceClassification, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label : i for i, label in enumerate(label_list,1)}
	
	features = []
	for (ex_index,example) in enumerate(examples):

		text_a, entity_a, entity_b = example.text_a.split('[RE]')

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
		if ex_index < 2:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(InputFeatures(
			input_ids=input_ids,
			input_mask=input_mask,
			segment_ids=segment_ids,
			label_id=label_id))
	return features

def create_train_test(positive_file_path,negative_file_path):
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
	positive_labels = ["Adverse effect"]*num_positive_sents
	num_negative_sents = len(negative_sents)
	negative_labels = ["Not an adverse effect"]*num_negative_sents

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

def train_and_evaluate(OUTPUT_DIR,do_train = True,do_eval=True):
	""" Train and evaluate a BERT NER Model"""

	
	BATCH_SIZE = 32
	LEARNING_RATE = 2e-5
	NUM_TRAIN_EPOCHS = 5.0

	#in this steps lr will be low and training will be slow
	WARMUP_PROPORTION = 0.1



	if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR) and do_train:
		raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
		
	#create train and test data

	train_sents,train_labels,test_sents,test_labels = create_train_test("ADE/DRUG-AE.rel","ADE/negative_data_AE.rel")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

	if do_train:

		train_examples = [InputExample(guid=None,text_a=sentence,text_b=None,label=label) for sentence,label in zip(train_sents, train_labels)]
		num_train_examples = len(train_examples)

		num_train_steps = int(math.ceil(num_train_examples / BATCH_SIZE * NUM_TRAIN_EPOCHS))
		num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

		model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = num_labels)
		model.to(device)

		param_optimizer = list(model.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]

		optimizer = BertAdam(optimizer_grouped_parameters,lr=LEARNING_RATE,warmup=WARMUP_PROPORTION,t_total=num_train_steps)

		global_step = 0
		nb_tr_steps = 0
		tr_loss = 0

		train_features = convert_examples_to_features(
			train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)


		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", num_train_examples)
		logger.info("  Batch size = %d", BATCH_SIZE)
		logger.info("  Num steps = %d", num_train_steps)


		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		train_sampler = RandomSampler(train_data)

		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

		model.train()
		# for name, param in model.named_parameters():
		#   if param.requires_grad:
		#       print(name)
		# return
		for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_id = batch
				loss = model(input_ids, segment_ids, input_mask, label_id)
				loss.backward()

				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				optimizer.step()
				optimizer.zero_grad()
				global_step += 1
			print(tr_loss)

		# Save a trained model and the associated configuration
		model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
		output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
		torch.save(model_to_save.state_dict(), output_model_file)
		output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
		with open(output_config_file, 'w') as f:
			f.write(model_to_save.config.to_json_string())
		label_map = {i : label for i, label in enumerate(label_list,1)}    
		model_config = {"bert_model":"bert-base-uncased","do_lower":True,"max_seq_length":MAX_SEQ_LENGTH,"num_labels":num_labels,"label_map":label_map}
		json.dump(model_config,open(os.path.join(OUTPUT_DIR,"model_config.json"),"w"))

	else:
		output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
		output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
		config = BertConfig(output_config_file)
		model = BertForSequenceClassification(config, num_labels=num_labels)
		model.load_state_dict(torch.load(output_model_file))

	model.to(device)

	if do_eval:

		EVAL_BATCH_SIZE = 32

		eval_examples = [InputExample(guid=None,text_a=sentence,text_b=None,label=label) for sentence,label in zip(test_sents, test_labels)]
		num_eval_examples = len(eval_examples)

		eval_features = convert_examples_to_features(
			eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", num_eval_examples)
		logger.info("  Batch size = %d", EVAL_BATCH_SIZE)
		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)        
		#   # Run prediction for full data
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
		model.eval()

		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		y_true = []
		y_pred = []
		label_map = {i : label for i, label in enumerate(label_list,1)}
		for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			label_ids = label_ids.to(device)

			with torch.no_grad():
				logits = model(input_ids, segment_ids, input_mask)
				
			logits = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
			logits = logits.detach().cpu().numpy()
			label_ids = label_ids.to('cpu').numpy()
			y_pred.extend(logits)
			y_true.extend(label_ids)
		print(len(y_pred))
		print(len(y_true))
		report = classification_report(y_true, y_pred)
		output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
		with open(output_eval_file, "w") as writer:
			logger.info("***** Eval results *****")
			logger.info("\n%s", report)
			writer.write(report)


def predict(OUTPUT_DIR,in_sentences):
	""" predict a bert model 
		OUTPUT_DIR :: contains pretrained models
		in_sentences :: is a list of sentences on which tagging has to be performed
	"""
	PRED_BATCH_SIZE = 64

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_config = os.path.join(OUTPUT_DIR,"model_config.json")
	model_config = json.load(open(model_config))
	output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
	output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
	config = BertConfig(output_config_file)
	model = BertForSequenceClassification(config, num_labels=model_config["num_labels"])
	model.load_state_dict(torch.load(output_model_file))
	model.to(device)
	tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"],do_lower_case=model_config["do_lower"])

	in_examples = [InputExample(guid="", text_a=x, text_b=None, label="Adverse effect") for x in in_sentences]
	in_features = convert_examples_to_features(in_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

	all_input_ids = torch.tensor([f.input_ids for f in in_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in in_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in in_features], dtype=torch.long)

	pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)       
	#   # Run prediction for full data
	pred_sampler = SequentialSampler(pred_data)
	pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=PRED_BATCH_SIZE,drop_last = False)
	model.eval()

	preds = []

	label_map = model_config["label_map"]

	for input_ids, input_mask, segment_ids in tqdm(pred_dataloader, desc="Predicting"):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)

		with torch.no_grad():
			logits = model(input_ids, segment_ids, input_mask)

		logits = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
		logits = logits.detach().cpu().numpy()  

		preds.extend(logits)
	label_map_reverse = {"1":"Adverse effect","2":"Not an adverse effect"}
	return [(sentence,label_map_reverse[str(pred)]) for sentence,pred in zip(in_sentences,preds)]

	
if __name__ == "__main__":
	# train_and_evaluate("pytorch_classifier_RE",do_train=True,do_eval=True)
	sents = ["treatment of philadelphia chromosome_positive acute lymphocytic leukemia with hyper_cvad and imatinib mesylate .[RE]imatinib[RE]leukemia"]
	print(predict("pytorch_classifier_RE",sents))

