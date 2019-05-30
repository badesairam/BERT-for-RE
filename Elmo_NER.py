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
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from allennlp.modules.elmo import Elmo, batch_to_ids



WEIGHTS_NAME = "pytorch_model.bin"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 128
label_list = ["B","I","O"]
num_labels = len(label_list)+1

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask,label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.label_id = label_id


class ElmoNER(nn.Module):
	def __init__(self,dim_elmo,hidden_size_lstm,dropout,num_labels):
		super().__init__()
		options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		self.elmo = Elmo(options_file,weight_file,3,dropout=0)
		self.dropout = nn.Dropout(p=dropout)
		self.word_lstm = nn.LSTM(dim_elmo, hidden_size_lstm//2, bidirectional=True)
		self.num_labels = num_labels
		self.hidden2tag = nn.Linear(hidden_size_lstm,num_labels)


	def forward(self,input,attention_mask=None,labels=None):
		word_emb = self.elmo(input)['elmo_representations'][1]
		word_emb = self.dropout(word_emb.transpose(0,1))
		lstm_out,_ = self.word_lstm(word_emb)
		lstm_out = self.dropout(lstm_out.transpose(0,1))
		logits = self.hidden2tag(lstm_out)
		loss_fct = nn.CrossEntropyLoss()
		if labels is not None:
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1,self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits,active_labels)
				return loss
		return logits

def convert_examples_to_features(examples, label_list, max_seq_length):
	""" Loads a data file into a list of InputFeatures"""

	label_map = {label : i for i, label in enumerate(label_list,1)}

	features = []
	for (ex_index,example) in enumerate(examples):
		input_words,label_syms = example
		if len(input_words)>=max_seq_length:
			input_words = input_words[:max_seq_length]
			label_syms = label_syms[:max_seq_length]

		label_ids = []
		for label in label_syms:
			label_ids.append(label_map[label])

		input_mask = [1]*len(input_words)

		input_words = [input_words]
		input_ids = batch_to_ids(input_words).numpy()

		input_ids = input_ids[0]
		input_pad = [[261]*50]

		while(len(label_ids)<max_seq_length):
			input_ids = np.append(input_ids,input_pad,axis=0)
			input_mask.append(0)
			label_ids.append(0)

		assert input_ids.shape[0] == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(label_ids) == max_seq_length

		if ex_index < 0:
			print(input_ids.shape)
			print(len(input_mask))
			print(len(label_ids))
			logger.info("*** Example ***")
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  label_id=label_ids))
	return features



def create_datasets(data_file):
	examples = []
	sentence = []
	label = []
	with open(data_file) as f:
		for line in f:
			if (len(line)==0) or line[0]=="\n":
				if len(sentence) > 0:
					examples.append((sentence,label))
					sentence = []
					label = []
				continue
			splits = [s.rstrip('\n').lstrip() for s in line.split(' ')]
			sentence.append(splits[0])
			label.append(splits[1])
		if len(sentence) > 0:
			examples.append((sentence,label))
			sentence = []
			label = []
	return examples,len(examples)

def train_and_evaluate(OUTPUT_DIR,train_file=None,test_file=None,do_train = True, do_eval = True):
	""" Train and evaluate a Elmo NER model"""
	BATCH_SIZE = 32
	LEARNING_RATE = 0.001
	NUM_TRAIN_EPOCHS = 5


	if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR) and do_train:
		raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if do_train:
		train_examples,num_train_examples = create_datasets(train_file)
		num_train_steps = int(math.ceil(num_train_examples / BATCH_SIZE * NUM_TRAIN_EPOCHS))

		train_features = convert_examples_to_features(train_examples,label_list,MAX_SEQ_LENGTH)

		#convert everything to tensors
		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


		train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
		train_sampler = RandomSampler(train_data)

		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

		model = ElmoNER(1024,512,0.5,num_labels)

		model.to(device)
		model.train()

		for name,param in model.named_parameters():
			if param.requires_grad:
				print(name)


		optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

		for _ in trange(int(NUM_TRAIN_EPOCHS),desc="Epoch"):
			tr_loss = 0
			for step,batch in enumerate(tqdm(train_dataloader,desc="Iteration")):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask,label_ids = batch
				loss = model(input_ids,input_mask,label_ids)
				loss.backward()

				tr_loss += loss.item()
				optimizer.step()
				optimizer.zero_grad()
			print(tr_loss)

		# Save a trained model and the associated configuration
		model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
		output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
		torch.save(model_to_save.state_dict(), output_model_file)
		label_map = {i : label for i, label in enumerate(label_list,1)}    
		model_config = {"dim_elmo":1024,"hidden_dim":512,"max_seq_length":MAX_SEQ_LENGTH,"num_labels":num_labels,"label_map":label_map}
		json.dump(model_config,open(os.path.join(OUTPUT_DIR,"model_config.json"),"w"))

	else:
		model_config = os.path.join(OUTPUT_DIR,"model_config.json")
		model_config = json.load(open(model_config))
		output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
		model = ElmoNER(model_config["dim_elmo"],model_config["hidden_dim"],0,model.config["num_labels"])
		model.load_state_dict(torch.load(output_model_file))

	if do_eval:
		EVAL_BATCH_SIZE = 32

		eval_examples , num_eval_examples = create_datasets(test_file)
		eval_features = convert_examples_to_features(eval_examples, label_list, MAX_SEQ_LENGTH,)
		
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", num_eval_examples)
		logger.info("  Batch size = %d", EVAL_BATCH_SIZE)
		
		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)		
		# 	# Run prediction for full data
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
		model.eval()

		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		y_true = []
		y_pred = []
		label_map = {i : label for i, label in enumerate(label_list,1)}
		for input_ids, input_mask,label_ids in tqdm(eval_dataloader, desc="Evaluating"):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			label_ids = label_ids.to(device)

			with torch.no_grad():
				logits = model(input_ids,input_mask)
				
			logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
			logits = logits.detach().cpu().numpy()
			label_ids = label_ids.to('cpu').numpy()
			input_mask = input_mask.to('cpu').numpy()
			for i,mask in enumerate(input_mask):
				temp_1 =  []
				temp_2 = []
				for j, m in enumerate(mask):
					if m:
						temp_1.append(label_map[label_ids[i][j]])
						temp_2.append(label_map[logits[i][j]])
					else:
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
	""" predict a elmo model 
		OUTPUT_DIR :: contains pretrained models
		in_sentences :: is a list of sentences on which tagging has to be performed
	"""
	PRED_BATCH_SIZE = 300

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	
	model_config = os.path.join(OUTPUT_DIR,"model_config.json")
	model_config = json.load(open(model_config))
	output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
	model = ElmoNER(model_config["dim_elmo"],model_config["hidden_dim"],0,model_config["num_labels"])
	model.load_state_dict(torch.load(output_model_file))

	model.to(device)

	in_examples = [(x.strip().split(" "),["O"]*len(x.strip().split(" "))) for x in in_sentences]
	in_features = convert_examples_to_features(in_examples, label_list, MAX_SEQ_LENGTH)

	all_input_ids = torch.tensor([f.input_ids for f in in_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in in_features], dtype=torch.long)

	pred_data = TensorDataset(all_input_ids, all_input_mask)		
	# 	# Run prediction for full data
	pred_sampler = SequentialSampler(pred_data)
	pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=PRED_BATCH_SIZE,drop_last = False)
	model.eval()

	print("model memory :: ",torch.cuda.memory_allocated(device=device))

	preds = []

	label_map = model_config["label_map"]

	for input_ids, input_mask in tqdm(pred_dataloader, desc="Predicting"):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)


		print("model memory with batch:: ",torch.cuda.memory_allocated(device=device))

		with torch.no_grad():
			logits = model(input_ids,input_mask)

		logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
		logits = logits.detach().cpu().numpy()	
		pred_batch = []
		for i,mask in enumerate(input_mask):
			temp_1 =  []
			for j, m in enumerate(mask):
				if m:
					temp_1.append(label_map[str(logits[i][j])])
				else:
					break
			pred_batch.append(temp_1)
		preds.extend(pred_batch)
	return [(sentence,pred) for sentence,pred in zip(in_sentences,preds)]


if __name__ == "__main__":
	# train_and_evaluate("Out_elmo",train_file="AGE/train.txt",test_file="AGE/valid.txt",do_train=True,do_eval=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("start memory :: ",torch.cuda.memory_allocated(device=device))
	sent = ["Her son is 35 years old ."]*2048
	predict("Out_elmo",sent)

	print("final memory :: ",torch.cuda.memory_allocated(device=device))

