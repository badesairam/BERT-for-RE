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
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
											  BertConfig,
											  BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
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
label_list = ["B","I","O", "X","[CLS]","[SEP]"]
num_labels = len(label_list)+1

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
		textlist = example.text_a.split(' ')
		textlist = list(filter(lambda a: a!='',textlist))
		labellist = example.label
		tokens = []
		labels = []
		for i, word in enumerate(textlist):
			token = tokenizer.tokenize(word)
			tokens.extend(token)
			label_1 = labellist[i]
			for m in range(len(token)):
				if m == 0:
					labels.append(label_1)
				else:
					labels.append("X")
		if len(tokens) >= max_seq_length - 1:
			tokens = tokens[0:(max_seq_length - 2)]
			labels = labels[0:(max_seq_length - 2)]
		ntokens = []
		segment_ids = []
		label_ids = []
		ntokens.append("[CLS]")
		segment_ids.append(0)
		label_ids.append(label_map["[CLS]"])
		for i, token in enumerate(tokens):
			ntokens.append(token)
			segment_ids.append(0)
			label_ids.append(label_map[labels[i]])
		ntokens.append("[SEP]")
		segment_ids.append(0)
		label_ids.append(label_map["[SEP]"])
		input_ids = tokenizer.convert_tokens_to_ids(ntokens)
		input_mask = [1] * len(input_ids)
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)
			label_ids.append(0)
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(label_ids) == max_seq_length
		
		if ex_index < 1:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_ids))
	return features

def create_datasets(data_file):
	examples = []
	sentence = ""
	label = []
	with open(data_file) as f:
		for line in f:
			if (len(line)==0) or line[0]=="\n":
				if len(sentence) > 0:
					examples.append(InputExample(guid=None, text_a=sentence, text_b=None, label=label))
					sentence = ""
					label = []
				continue
			splits = [s.rstrip('\n').lstrip() for s in line.split(' ')]
			sentence = sentence +" "+splits[0]
			sentence.strip()
			label.append(splits[1])
		if len(sentence) > 0:
			examples.append(InputExample(guid=None, text_a=sentence, text_b=None, label=label))
			sentence = ""
			label = []
	return examples,len(examples)


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
		

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

	if do_train:
		train_examples, num_train_examples = create_datasets("AGE/train.txt")

		num_train_steps = int(math.ceil(num_train_examples / BATCH_SIZE * NUM_TRAIN_EPOCHS))
		num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

		model = BertForTokenClassification.from_pretrained("bert-base-uncased",num_labels = num_labels)
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
		# 	if param.requires_grad:
		# 		print(name)
		# return
		for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids = batch
				loss = model(input_ids, segment_ids, input_mask, label_ids)
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
		model_config = {"bert_model":"bert-base-uncased","do_lower":True,"max_seq_length":MAX_SEQ_LENGTH,"num_labels":len(label_list)+1,"label_map":label_map}
		json.dump(model_config,open(os.path.join(OUTPUT_DIR,"model_config.json"),"w"))

	else:
		output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
		output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
		config = BertConfig(output_config_file)
		model = BertForTokenClassification(config, num_labels=num_labels)
		model.load_state_dict(torch.load(output_model_file))

	model.to(device)

	if do_eval:

		EVAL_BATCH_SIZE = 32

		eval_examples , num_eval_examples = create_datasets("AGE/valid.txt")
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
		# 	# Run prediction for full data
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
				
			logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
			logits = logits.detach().cpu().numpy()
			label_ids = label_ids.to('cpu').numpy()
			input_mask = input_mask.to('cpu').numpy()
			for i,mask in enumerate(input_mask):
				temp_1 =  []
				temp_2 = []
				for j, m in enumerate(mask):
					if j == 0:
						continue
					if m:
						if label_map[label_ids[i][j]] != "X":
							temp_1.append(label_map[label_ids[i][j]])
							temp_2.append(label_map[logits[i][j]])
					else:
						temp_1.pop()
						temp_2.pop()
						break
				y_true.append(temp_1)
				y_pred.append(temp_2)
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
	model = BertForTokenClassification(config, num_labels=model_config["num_labels"])
	model.load_state_dict(torch.load(output_model_file))
	model.to(device)
	tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"],do_lower_case=model_config["do_lower"])

	in_examples = [InputExample(guid="", text_a=x, text_b=None, label=["O"]*len(x.split(" "))) for x in in_sentences]
	in_features = convert_examples_to_features(in_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

	all_input_ids = torch.tensor([f.input_ids for f in in_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in in_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in in_features], dtype=torch.long)

	pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)		
	# 	# Run prediction for full data
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

		logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
		logits = logits.detach().cpu().numpy()	
		pred_batch = []
		for i,mask in enumerate(input_mask):
			temp_1 =  []
			for j, m in enumerate(mask):
				if j == 0:
					continue
				if m:
					if label_map[str(logits[i][j])] != "X":
						temp_1.append(label_map[str(logits[i][j])])
				else:
					temp_1.pop()
					break
			pred_batch.append(temp_1)
		preds.extend(pred_batch)
	return [(sentence,pred) for sentence,pred in zip(in_sentences,preds)]
if __name__ == "__main__":
	train_and_evaluate("tmp",do_train=True,do_eval=True)
	sent = ["Her son is 35 years old ."]*2
	# print(predict("Out_BERT",sent))

