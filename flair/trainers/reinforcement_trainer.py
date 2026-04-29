"""
Fine-tune trainer: a trainer for finetuning BERT and able to be parallelized based on flair
Author: Xinyu Wang
Contact: wangxy1@shanghaitech.edu.cn
"""

from .distillation_trainer import *
from transformers import (
	AdamW,
	get_linear_schedule_with_warmup,
)
from flair.models.biaffine_attention import BiaffineAttention, BiaffineFunction
# from flair.models.dependency_model import generate_tree, convert_score_back
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import random
import copy
from flair.parser.utils.alg import crf
import h5py
from flair.models.controller import EmbedController
import numpy as np 
import json

import gc

# def check_garbage():
#   for obj in gc.get_objects():
#       try:
#           if torch.is_tensor(obj):
#               pring(type(obj),obj.size())
#       except:
#           pass


def count_parameters(model):
	total_param = 0
	for name,param in model.named_parameters():
		num_param = np.prod(param.size())
		# print(name,num_param)
		total_param+=num_param
	return total_param

dependency_tasks={'enhancedud', 'dependency', 'srl', 'ner_dp'}
def get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, fix_embedding_steps, steepness = 0.5, factor = 5, model_size=1, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases linearly after
	linearly increasing during a warmup period.
	"""

	def lr_lambda(current_step):
		# step 0 ~ fix_embedding_steps: no modification
		# step fix_embedding_steps ~ num_warmup_steps + fix_embedding_steps: warmup embedding training
		# step num_warmup_steps + fix_embedding_steps ~ : square root decay
		if current_step < fix_embedding_steps:
			return 1
		elif current_step < num_warmup_steps + fix_embedding_steps:
			return float(current_step-fix_embedding_steps) / float(max(1, num_warmup_steps))
		step = max(current_step - num_warmup_steps - fix_embedding_steps, 1)
		return max(0.0, factor * (model_size ** (-0.5) * min(step ** (-steepness), step * num_warmup_steps ** (-steepness - 1))))

	return LambdaLR(optimizer, lr_lambda, last_epoch)

class ReinforcementTrainer(ModelDistiller):
	def __init__(
		self,
		model: flair.nn.Model,
		teachers, # None, for consistency with other trainers
		corpus: ListCorpus,
		optimizer = AdamW,
		controller_optimizer = Adam,
		controller_learning_rate: float = 0.1,
		epoch: int = 0,
		distill_mode = False,
		optimizer_state: dict = None,
		scheduler_state: dict = None,
		use_tensorboard: bool = False,
		language_resample = False,
		config = None,
		is_test: bool = False,
		direct_upsample_rate: int = -1,
		down_sample_amount: int = -1,
		sentence_level_batch: bool = False,
		dev_sample: bool = False,
		assign_doc_id: bool = False,
		train_with_doc: bool = False,
		pretrained_file_dict: dict = {},
		sentence_level_pretrained_data: bool = False,
		# **kwargs,
	):
		"""
		Initialize a model trainer
		:param model: The model that you want to train. The model should inherit from flair.nn.Model
		:param corpus: The dataset used to train the model, should be of type Corpus
		:param optimizer: The optimizer to use (Default AdamW for finetuning BERT)
		:param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
		:param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
		:param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
		:param use_tensorboard: If True, writes out tensorboard information
		"""
		# if teachers is not None:
		#   assert len(teachers)==len(corpus.train_list), 'Training data and teachers should be the same length now!'
		self.model: flair.nn.Model = model
		# Grouped-ACE: extract num_groups, group_seed, group_mode before passing to EmbedController
		_controller_config = {k: v for k, v in config['Controller'].items()}
		self.num_groups = int(_controller_config.pop('num_groups', 1))
		group_seed = _controller_config.pop('group_seed', None)
		group_mode = _controller_config.pop('group_mode', 'equal')          # 'equal' | 'golden' | 'entity_golden'
		golden_dims_file = _controller_config.pop('golden_dims_file', None)  # path to JSON

		# Entity-Golden specific params (popped so they don't reach EmbedController)
		entities_per_group = int(_controller_config.pop('entities_per_group', 1))
		allow_duplicate_dims = bool(_controller_config.pop('allow_duplicate_dims', False))
		entity_merge_map = _controller_config.pop('entity_merge_map', None)  # optional manual merge

		# Constraint: ensure each entity group has ≥1 active embedding per episode
		ensure_entity_coverage = bool(_controller_config.pop('ensure_entity_coverage', False))

		# Runtime K: slice stored top-N dims to top-K at runtime (store_top_n mode)
		runtime_max_golden_dims = int(_controller_config.pop('max_golden_dims', 0))

		# Golden-dims compute params — used only by compute script, not by EmbedController
		for _gdkey in ('selection_mode', 'mad_k', 'percentile',
		               'snr_ratio', 'elbow_min_k', 'elbow_max_k', 'threshold'):
			_controller_config.pop(_gdkey, None)

		# Build per-embedding permutation indices
		# perm_indices[i] = LongTensor of dim indices for embedding i, shape [embed_dim]
		# After applying perm, torch.chunk splits into groups:
		#   equal mode  → random permutation → groups are random subsets
		#   golden mode → d-score-sorted perm → group 0 = entity dims, group N-1 = O dims
		perm_indices = None
		group_masks = None   # Entity-Golden: list[list[FloatTensor]], group_masks[emb][grp] = (D,)
		if self.num_groups > 1 or group_mode == 'entity_golden':
			# IMPORTANT: must follow the same order used later in sequence_tagger_model
			# (sorted(sentences.features.keys()) == sorted([embedding.name ...]))
			emb_name_dim = sorted([(e.name, e.embedding_length) for e in self.model.embeddings.embeddings], key=lambda x: x[0])
			emb_names = [x[0] for x in emb_name_dim]
			embed_dims = [x[1] for x in emb_name_dim]

			if group_mode == 'entity_golden' and golden_dims_file is not None:
				# --- Entity-Golden mode: per-entity-type dim masks ---
				group_masks = self._build_entity_golden_masks(
					golden_dims_file, emb_names, embed_dims,
					config.get('embeddings', {}),
					entities_per_group, allow_duplicate_dims, entity_merge_map,
					runtime_max_golden_dims,
				)
				if group_masks is not None:
					# num_groups is determined by the merge result
					self.num_groups = len(group_masks[0])
					log.info(f'Entity-Golden ACE: {self.num_groups} groups/embedding, '
							 f'allow_duplicate_dims={allow_duplicate_dims}')

			elif group_mode == 'golden' and golden_dims_file is not None:
				# --- Golden mode: load pre-computed d-score sorted indices ---
				import json
				with open(golden_dims_file, encoding='utf-8') as _f:
					golden_data = json.load(_f)
				perm_indices = []
				# Match runtime names → config keys by content (not index)
				_runtime_to_cfg = self._build_runtime_to_config_map(emb_names, config.get('embeddings', {}))
				for i, emb_name in enumerate(emb_names):
					entry = None
					if emb_name in golden_data:
						entry = golden_data[emb_name]
					elif emb_name in _runtime_to_cfg and _runtime_to_cfg[emb_name] in golden_data:
						entry = golden_data[_runtime_to_cfg[emb_name]]

					if entry is not None:
						group_key = f'N{self.num_groups}'
						if group_key in entry.get('groups', {}):
							# Flatten groups back to a single sorted permutation
							all_indices = []
							for grp in entry['groups'][group_key]:
								all_indices.extend(grp)
							perm_indices.append(torch.tensor(all_indices, dtype=torch.long))
						else:
							# Fallback: use plain sorted_indices if exact N not pre-computed
							perm_indices.append(torch.tensor(entry['sorted_indices'], dtype=torch.long))
					else:
						log.warning(f'Golden dims not found for {emb_name}, falling back to sequential split')
						perm_indices.append(torch.arange(embed_dims[i]))
				if len(perm_indices) != len(embed_dims):
					log.warning('Golden dims count mismatch — falling back to sequential split')
					perm_indices = None
				else:
					log.info(f'Golden-ACE: loaded pre-computed d-score permutations from {golden_dims_file}')

			elif group_mode == 'equal' and group_seed is not None:
				# --- Equal mode: random permutation with fixed seed ---
				perm_indices = []
				rng = torch.Generator()
				rng.manual_seed(int(group_seed))
				for dim in embed_dims:
					perm_indices.append(torch.randperm(dim, generator=rng))
			# else: perm_indices stays None → sequential split (dims 0..D/N, D/N..2D/N, ...)

		self.model.num_groups = self.num_groups
		self.model.group_perm_indices = perm_indices
		self.model.group_masks = group_masks
		self.ensure_entity_coverage = ensure_entity_coverage
		self.runtime_max_golden_dims = runtime_max_golden_dims
		self.controller: flair.nn.Model = EmbedController(num_actions=len(self.model.embeddings.embeddings) * self.num_groups, state_size = self.model.embeddings.embedding_length, **_controller_config)
		self.model.use_rl = True
		if self.controller.model_structure is not None:
			self.model.use_embedding_masks = True
		self.model.embedding_selector = True
		self.corpus: ListCorpus = corpus
		# num_languages = len(self.corpus.targets)
		self.controller_learning_rate=controller_learning_rate
		self.corpus2id = {x:i for i,x in enumerate(self.corpus.targets)}
		self.id2corpus = {i:x for i,x in enumerate(self.corpus.targets)}
		self.sentence_level_batch = sentence_level_batch
		if language_resample or direct_upsample_rate>0:
			sent_per_set=torch.FloatTensor([len(x) for x in self.corpus.train_list])
			total_sents=sent_per_set.sum()
			sent_each_dataset=sent_per_set/total_sents
			exp_sent_each_dataset=sent_each_dataset.pow(0.7)
			sent_sample_prob=exp_sent_each_dataset/exp_sent_each_dataset.sum()
		self.sentence_level_pretrained_data=sentence_level_pretrained_data

		if assign_doc_id:
			doc_sentence_dict = {}
			same_corpus_mapping = {'CONLL_06_GERMAN': 'CONLL_03_GERMAN_NEW',
			'CONLL_03_GERMAN_DP': 'CONLL_03_GERMAN_NEW',
			'CONLL_03_DP': 'CONLL_03_ENGLISH',
			'CONLL_03_DUTCH_DP': 'CONLL_03_DUTCH_NEW',
			'CONLL_03_SPANISH_DP': 'CONLL_03_SPANISH_NEW'}
			for corpus_id in range(len(self.corpus2id)):
				
				if self.corpus.targets[corpus_id] in same_corpus_mapping:
					corpus_name = same_corpus_mapping[self.corpus.targets[corpus_id]].lower()+'_'
				else:
					corpus_name = self.corpus.targets[corpus_id].lower()+'_'
				doc_sentence_dict = self.assign_documents(self.corpus.train_list[corpus_id], 'train_', doc_sentence_dict, corpus_name, train_with_doc)
				doc_sentence_dict = self.assign_documents(self.corpus.dev_list[corpus_id], 'dev_', doc_sentence_dict, corpus_name, train_with_doc)
				doc_sentence_dict = self.assign_documents(self.corpus.test_list[corpus_id], 'test_', doc_sentence_dict, corpus_name, train_with_doc)
				if train_with_doc:
					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.train_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.train_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.train_list[corpus_id].reset_sentence_count

					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.dev_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.dev_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.dev_list[corpus_id].reset_sentence_count

					new_sentences=[]
					for sentid, sentence in enumerate(self.corpus.test_list[corpus_id]):
						if sentence[0].text=='-DOCSTART-':
							continue
						new_sentences.append(sentence)
					self.corpus.test_list[corpus_id].sentences = new_sentences.copy()
					self.corpus.test_list[corpus_id].reset_sentence_count

			if train_with_doc:
				self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
				self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])		
				self.corpus._test: FlairDataset = ConcatDataset([data for data in self.corpus.test_list])		
			# for key in pretrained_file_dict:
			# pdb.set_trace()
			for embedding in self.model.embeddings.embeddings:
				if embedding.name in pretrained_file_dict:
					self.assign_predicted_embeddings(doc_sentence_dict,embedding,pretrained_file_dict[embedding.name])

		for corpus_name in self.corpus2id:
			i = self.corpus2id[corpus_name]
			for sentence in self.corpus.train_list[i]:
				sentence.lang_id=i
			if len(self.corpus.dev_list)>i:
				for sentence in self.corpus.dev_list[i]:
					sentence.lang_id=i
			if len(self.corpus.test_list)>i:
				for sentence in self.corpus.test_list[i]:
					sentence.lang_id=i
			if language_resample:
				length = len(self.corpus.train_list[i])
				# idx = random.sample(range(length), int(sent_sample_prob[i] * total_sents))
				idx = torch.randint(length, (int(sent_sample_prob[i] * total_sents),))
				self.corpus.train_list[i].sentences = [self.corpus.train_list[i][x] for x in idx]
			if direct_upsample_rate>0:
				if len(self.corpus.train_list[i].sentences)<(sent_per_set.max()/direct_upsample_rate).item():
					res_sent=[]
					dev_res_sent=[]
					for sent_batch in range(direct_upsample_rate):
						res_sent+=copy.deepcopy(self.corpus.train_list[i].sentences)
						if config['train']['train_with_dev']:
							dev_res_sent+=copy.deepcopy(self.corpus.dev_list[i].sentences)
					self.corpus.train_list[i].sentences = res_sent
					self.corpus.train_list[i].reset_sentence_count
					if config['train']['train_with_dev']:
						self.corpus.dev_list[i].sentences = dev_res_sent
						self.corpus.dev_list[i].reset_sentence_count
			if down_sample_amount>0:
				if len(self.corpus.train_list[i].sentences)>down_sample_amount:
					self.corpus.train_list[i].sentences = self.corpus.train_list[i].sentences[:down_sample_amount]
					self.corpus.train_list[i].reset_sentence_count
					if config['train']['train_with_dev']:
						self.corpus.dev_list[i].sentences = self.corpus.dev_list[i].sentences[:down_sample_amount]
						self.corpus.dev_list[i].reset_sentence_count
					if dev_sample:
						self.corpus.dev_list[i].sentences = self.corpus.dev_list[i].sentences[:down_sample_amount]
						self.corpus.dev_list[i].reset_sentence_count
		if direct_upsample_rate>0 or down_sample_amount:
			self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
			if config['train']['train_with_dev']:
				self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])
		print(self.corpus)

		# self.corpus = self.assign_pretrained_teacher_predictions(self.corpus,self.corpus_teacher,self.teachers)
		self.update_params_group=[]
		
		self.optimizer: torch.optim.Optimizer = optimizer
		if type(optimizer)==str:
			self.optimizer = getattr(torch.optim,optimizer)

		self.controller_optimizer: torch.optim.Optimizer = controller_optimizer
		if type(controller_optimizer) == str:
			self.controller_optimizer = getattr(torch.optim,controller_optimizer)



		self.epoch: int = epoch
		self.scheduler_state: dict = scheduler_state
		self.optimizer_state: dict = optimizer_state
		self.use_tensorboard: bool = use_tensorboard
		
		self.config = config
		self.use_bert = False
		self.bert_tokenizer = None
		for embedding in self.model.embeddings.embeddings:
			if 'bert' in embedding.__class__.__name__.lower():
				self.use_bert=True
				self.bert_tokenizer = embedding.tokenizer

		if hasattr(self.model,'remove_x') and self.model.remove_x:
			for corpus_id in range(len(self.corpus2id)):
				for sent_id, sentence in enumerate(self.corpus.train_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
				for sent_id, sentence in enumerate(self.corpus.dev_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
				for sent_id, sentence in enumerate(self.corpus.test_list[corpus_id]):
					sentence.orig_sent=copy.deepcopy(sentence)
					words = [x.text for x in sentence.tokens]
					if '<EOS>' in words:
						eos_id = words.index('<EOS>')
						sentence.chunk_sentence(0,eos_id)
					else:
						pass
			self.corpus._train: FlairDataset = ConcatDataset([data for data in self.corpus.train_list])
			self.corpus._dev: FlairDataset = ConcatDataset([data for data in self.corpus.dev_list])		
			self.corpus._test: FlairDataset = ConcatDataset([data for data in self.corpus.test_list])		


	# ------------------------------------------------------------------
	# Entity-Golden ACE: build per-group dim masks
	# ------------------------------------------------------------------
	def _build_entity_golden_masks(
		self,
		golden_dims_file: str,
		emb_names: list,
		embed_dims: list,
		emb_config: dict,
		entities_per_group: int = 1,
		allow_duplicate_dims: bool = False,
		entity_merge_map: list = None,
		runtime_max_golden_dims: int = 0,
	):
		"""
		Build group_masks for Entity-Golden mode.

		Returns:
		  group_masks: list[list[FloatTensor]]
		    group_masks[emb_idx][group_id] = FloatTensor of size D
		    1.0 at dim positions that belong to this group, 0.0 otherwise.

		Each group corresponds to one (or merged) entity type.
		With allow_duplicate_dims=False, overlapping dims go to the entity
		with highest |d|.  With True, they appear in all qualifying groups.
		Remainder dims (not golden for any entity) are excluded (always 0).
		"""
		import json

		with open(golden_dims_file, encoding='utf-8') as _f:
			golden_data = json.load(_f)

		# Resolve embedding name → JSON key mapping (content-based, not index-based)
		_runtime_to_cfg = self._build_runtime_to_config_map(emb_names, emb_config)

		all_entity_types = golden_data.get('_meta', {}).get('entity_types', None)

		group_masks = []

		for i, emb_name in enumerate(emb_names):
			D = embed_dims[i]
			# Find the JSON entry for this embedding
			entry = None
			if emb_name in golden_data:
				entry = golden_data[emb_name]
			elif emb_name in _runtime_to_cfg and _runtime_to_cfg[emb_name] in golden_data:
				entry = golden_data[_runtime_to_cfg[emb_name]]

			if entry is None:
				log.warning(f'Entity-Golden: no data for {emb_name}, all dims will be zero')
				# Create single empty group to avoid crash
				group_masks.append([torch.zeros(D)])
				continue

			entity_types = entry.get('entity_types', all_entity_types or [])
			raw_golden = entry.get('golden_dims_per_entity', {})
			# --- Runtime K slicing ---
			# If runtime_max_golden_dims > 0, use only the first K stored dims per entity.
			# This allows storing top-N (large) in the JSON and slicing at runtime.
			if runtime_max_golden_dims > 0:
				golden_per_entity = {et: dims[:runtime_max_golden_dims] for et, dims in raw_golden.items()}
				log.info(f'  {emb_name}: runtime_max_golden_dims={runtime_max_golden_dims} '
					     f'(stored={max((len(v) for v in raw_golden.values()), default=0)})')
			else:
				golden_per_entity = raw_golden
			d_scores_per_entity = entry.get('per_entity_d_scores', {})

			# --- Determine entity groups (merge if needed) ---
			entity_groups = self._merge_entities(
				entity_types, entities_per_group, entity_merge_map
			)

			n_groups = len(entity_groups)

			# --- Build per-group golden dim sets ---
			# group_golden[g] = set of dim indices
			group_golden = []
			for g, ent_list in enumerate(entity_groups):
				dims = set()
				for etype in ent_list:
					if etype in golden_per_entity:
						dims.update(golden_per_entity[etype])
				group_golden.append(dims)

			# --- Handle overlapping dims ---
			if not allow_duplicate_dims:
				# For each dim that appears in >1 group, keep it only in
				# the group whose entity has the highest |d| for that dim.
				all_dims = set()
				for g_dims in group_golden:
					all_dims.update(g_dims)

				for dim in all_dims:
					# Which groups contain this dim?
					containing = [g for g, gd in enumerate(group_golden) if dim in gd]
					if len(containing) <= 1:
						continue
					# Find best group: entity with max |d| at this dim
					best_g = containing[0]
					best_d = 0.0
					for g in containing:
						for etype in entity_groups[g]:
							if etype in d_scores_per_entity:
								d_val = abs(d_scores_per_entity[etype][dim])
								if d_val > best_d:
									best_d = d_val
									best_g = g
					# Remove from all except best
					for g in containing:
						if g != best_g:
							group_golden[g].discard(dim)

			# --- Convert to FloatTensor masks ---
			masks = []
			for g in range(n_groups):
				mask = torch.zeros(D)
				for dim in group_golden[g]:
					if dim < D:
						mask[dim] = 1.0
				masks.append(mask)

			# Log stats
			total_golden = sum(m.sum().item() for m in masks)
			log.info(f'  {emb_name} (D={D}): {n_groups} groups, '
					 f'{int(total_golden)} golden dims ({100*total_golden/D:.1f}%)')
			for g, (m, ent_list) in enumerate(zip(masks, entity_groups)):
				log.info(f'    Group {g} [{"+".join(ent_list)}]: {int(m.sum().item())} dims')

			group_masks.append(masks)

		# Validate: all embeddings must have same number of groups
		n_groups_list = [len(m) for m in group_masks]
		if len(set(n_groups_list)) > 1:
			log.error(f'Entity-Golden: inconsistent group counts across embeddings: {n_groups_list}')
			return None

		return group_masks

	@staticmethod
	def _merge_entities(entity_types, entities_per_group, entity_merge_map=None):
		"""
		Merge entity types into groups.

		Returns: list of lists, e.g. [['GPE','LOCATION-UNK'], ['ORG','FACILITY'], ...]

		If entity_merge_map is provided (list of lists), use it directly.
		Otherwise, auto-merge by pairing entities in sorted order.
		entities_per_group=1 means no merging (one entity per group).
		"""
		if entity_merge_map is not None:
			# Manual merge: validate and use directly
			return [list(group) for group in entity_merge_map]

		if entities_per_group <= 1:
			return [[e] for e in entity_types]

		# Auto-merge: simple round-robin pairing in sorted order
		n_entities = len(entity_types)
		n_groups = max(1, -(-n_entities // entities_per_group))  # ceil division
		groups = [[] for _ in range(n_groups)]
		for i, etype in enumerate(entity_types):
			groups[i % n_groups].append(etype)
		return groups

	@staticmethod
	def _build_runtime_to_config_map(emb_names, emb_config):
		"""Map runtime embedding names to config keys by matching config content.

		Runtime names (sorted alphabetically) and config keys (sorted alphabetically)
		have different orderings. This method matches them by examining the config
		values (model path, embeddings path, language, etc.).
		"""
		mapping = {}
		used_cfg = set()
		for rname in emb_names:
			for cfg_key, cfg_val in emb_config.items():
				if cfg_key in used_cfg:
					continue
				if not isinstance(cfg_val, dict):
					continue
				# TransformerWordEmbeddings / WordEmbeddings: model or embeddings path
				model_path = cfg_val.get('model', cfg_val.get('embeddings', ''))
				if model_path and (model_path in rname or rname in model_path):
					mapping[rname] = cfg_key
					used_cfg.add(cfg_key)
					break
				# BytePairEmbeddings: language string in runtime name
				lang = cfg_val.get('language', '')
				if lang and 'BytePair' in cfg_key and f'bpe-{lang}' in rname:
					mapping[rname] = cfg_key
					used_cfg.add(cfg_key)
					break
				# FastCharacterEmbeddings → runtime name 'Char'
				if 'Char' in cfg_key and rname == 'Char':
					mapping[rname] = cfg_key
					used_cfg.add(cfg_key)
					break
		if mapping:
			pairs = ', '.join(f'{r} → {c}' for r, c in mapping.items())
			log.info(f'Runtime→Config mapping: {pairs}')
		return mapping


	def train(
		self,
		base_path: Union[Path, str],
		learning_rate: float = 5e-5,
		mini_batch_size: int = 32,
		eval_mini_batch_size: int = None,
		max_epochs: int = 100,
		max_episodes: int = 10,
		seed: int = None,
		anneal_factor: float = 0.5,
		patience: int = 10,
		min_learning_rate: float = 5e-9,
		train_with_dev: bool = False,
		macro_avg: bool = True,
		monitor_train: bool = False,
		monitor_test: bool = False,
		embeddings_storage_mode: str = "cpu",
		checkpoint: bool = False,
		save_final_model: bool = True,
		anneal_with_restarts: bool = False,
		shuffle: bool = True,
		true_reshuffle: bool = False,
		param_selection_mode: bool = False,
		num_workers: int = 4,
		sampler=None,
		use_amp: bool = False,
		amp_opt_level: str = "O1",
		max_epochs_without_improvement = 30,
		warmup_steps: int = 0,
		use_warmup: bool = True,
		gradient_accumulation_steps: int = 1,
		lr_rate: int = 1,
		decay: float = 0.75,
		decay_steps: int = 5000,
		sort_data: bool = True,
		fine_tune_mode: bool = False,
		debug: bool = False,
		min_freq: int = -1,
		min_lemma_freq: int = -1,
		min_pos_freq: int = -1,
		rootschedule: bool = False,
		freezing: bool = False,
		log_reward: bool = False,
		sqrt_reward: bool = False,
		controller_momentum: float = 0.0,
		discount: float = 0.5,
		curriculum_file = None,
		random_search = False,
		continue_training = False,
		old_reward = False,
		**kwargs,
	) -> dict:

		"""
		Trains any class that implements the flair.nn.Model interface.
		:param base_path: Main path to which all output during training is logged and models are saved
		:param learning_rate: Initial learning rate
		:param mini_batch_size: Size of mini-batches during training
		:param eval_mini_batch_size: Size of mini-batches during evaluation
		:param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
		:param anneal_factor: The factor by which the learning rate is annealed
		:param patience: Patience is the number of epochs with no improvement the Trainer waits
		 until annealing the learning rate
		:param min_learning_rate: If the learning rate falls below this threshold, training terminates
		:param train_with_dev: If True, training is performed using both train+dev data
		:param monitor_train: If True, training data is evaluated at end of each epoch
		:param monitor_test: If True, test data is evaluated at end of each epoch
		:param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
		'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
		:param checkpoint: If True, a full checkpoint is saved at end of each epoch
		:param save_final_model: If True, final model is saved
		:param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
		:param shuffle: If True, data is shuffled during training
		:param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
		parameter selection.
		:param num_workers: Number of workers in your data loader.
		:param sampler: You can pass a data sampler here for special sampling of data.
		:param kwargs: Other arguments for the Optimizer
		:return:
		"""
		self.n_gpu = torch.cuda.device_count()
		# --- Global seed for reproducibility ---
		if seed is not None:
			import random, numpy as np
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			np.random.seed(seed)
			random.seed(seed)
			log.info(f'Global seed set to {seed}')
		default_learning_rate = learning_rate
		self.embeddings_storage_mode=embeddings_storage_mode
		self.mini_batch_size=mini_batch_size
		if self.use_tensorboard:
			try:
				from torch.utils.tensorboard import SummaryWriter

				writer = SummaryWriter()
			except:
				log_line(log)
				log.warning(
					"ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
				)
				log_line(log)
				self.use_tensorboard = False
				pass

		if use_amp:
			if sys.version_info < (3, 0):
				raise RuntimeError("Apex currently only supports Python 3. Aborting.")
			if amp is None:
				raise RuntimeError(
					"Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
					"to enable mixed-precision training."
				)

		if eval_mini_batch_size is None:
			eval_mini_batch_size = mini_batch_size

		# cast string to Path
		if type(base_path) is str:
			base_path = Path(base_path)

		log_handler = add_file_handler(log, base_path / "training.log")
		
		log_line(log)
		log.info(f'Model: "{self.model}"')
		log_line(log)
		log.info(f'Corpus: "{self.corpus}"')
		log_line(log)
		log.info("Parameters:")
		log.info(f' - Optimizer: "{self.optimizer.__name__}"')
		log.info(f' - learning_rate: "{learning_rate}"')
		log.info(f' - mini_batch_size: "{mini_batch_size}"')
		log.info(f' - patience: "{patience}"')
		log.info(f' - anneal_factor: "{anneal_factor}"')
		log.info(f' - max_epochs: "{max_epochs}"')
		log.info(f' - shuffle: "{shuffle}"')
		log.info(f' - train_with_dev: "{train_with_dev}"')
		log.info(f' - word min_freq: "{min_freq}"')
		log_line(log)
		log.info(f'Model training base path: "{base_path}"')
		log_line(log)
		log.info(f"Device: {flair.device}")
		log_line(log)
		log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

		# determine what splits (train, dev, test) to evaluate and log
		if monitor_train:
			assert 0, 'monitor_train is not supported now!'            
		# if train_with_dev:
		#   assert 0, 'train_with_dev is not supported now!'

		log_train = True if monitor_train else False
		log_test = (
			True
			if (not param_selection_mode and self.corpus.test and monitor_test)
			else False
		)
		log_dev = True if not train_with_dev else False

		# prepare loss logging file and set up header
		loss_txt = init_output_file(base_path, "loss.tsv")



		if self.controller_optimizer.__name__ == 'SGD':
			controller_optimizer: torch.optim.Optimizer = self.controller_optimizer(self.controller.parameters(), lr=self.controller_learning_rate, momentum=controller_momentum)
		else:
			controller_optimizer: torch.optim.Optimizer = self.controller_optimizer(self.controller.parameters(), lr=self.controller_learning_rate)
		
		
		if continue_training:
			# though this is not the final model of training, but we use this currently to save the space
			if (base_path / "best-model.pt").exists():
				self.model = self.model.load(base_path / "best-model.pt")
			self.controller = self.controller.load(base_path / "controller.pt")
			if (base_path/'controller_optimizer_state.pt').exists():
				controller_optimizer.load_state_dict(torch.load(base_path/'controller_optimizer_state.pt'))
			training_state = torch.load(base_path/'training_state.pt')
			start_episode = training_state.get('episode', 0)
			self.best_action = training_state.get('best_action', None)
			self.action_dict = training_state.get('action_dict', {})
			baseline_score = training_state.get('baseline_score', 0)
			# Restore grouped state in a backward-compatible way for old/plain checkpoints.
			if 'num_groups' in training_state:
				self.model.num_groups = training_state.get('num_groups', getattr(self.model, 'num_groups', 1))
			if 'group_perm_indices' in training_state:
				self.model.group_perm_indices = training_state.get('group_perm_indices', None)
			if 'group_masks' in training_state and training_state['group_masks'] is not None:
				self.model.group_masks = training_state['group_masks']
			# pdb.set_trace()
		else:
			start_episode=0
			self.action_dict = {}
			baseline_score=0
		# weight_extractor = WeightExtractor(base_path)
		# finetune_params = {name:param for name,param in self.model.named_parameters()}
		finetune_params=[param for name,param in self.model.named_parameters() if 'embedding' in name or name=='linear.weight' or name=='linear.bias']
		other_params=[param for name,param in self.model.named_parameters() if 'embedding' not in name and name !='linear.weight' and name !='linear.bias']
		# other_params = {name:param for name,param in self.model.named_parameters() if 'embeddings' not in name}
		

		# minimize training loss if training with dev data, else maximize dev score
		
		# start from here, the train data is a list now


		train_data = self.corpus.train_list
		if train_with_dev:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
		batch_loader=ColumnDataLoader(ConcatDataset(train_data),mini_batch_size,shuffle,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch)
		batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
		
		# Initialize dev and test loaders based on configuration
		if train_with_dev:
			# When training with dev data, we need to create loaders for evaluation
			if hasattr(self.corpus, 'dev_list') and hasattr(self.corpus, 'test_list'):
				# Multi-corpus setup (macro_avg style)
				dev_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.dev_list]
				for loader in dev_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)

				test_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.test_list]
				for loader in test_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				
				# Set macro_avg flag for this case
				macro_avg = True
			else:
				# Single corpus setup
				dev_loader=ColumnDataLoader(list(self.corpus.dev),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				dev_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				test_loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				test_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				
				# Set macro_avg flag for this case
				macro_avg = False
		else:
			# Standard evaluation setup
			if hasattr(self.corpus, 'dev_list') and hasattr(self.corpus, 'test_list'):
				# Multi-corpus setup (macro_avg style)
				dev_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.dev_list]
				for loader in dev_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)

				test_loaders=[ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch) \
							 for subcorpus in self.corpus.test_list]
				for loader in test_loaders:
					loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				
				# Set macro_avg flag for this case
				macro_avg = True
			else:
				# Single corpus setup
				dev_loader=ColumnDataLoader(list(self.corpus.dev),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				dev_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				test_loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, sort_data=sort_data, model = self.model, sentence_level_batch = self.sentence_level_batch)
				test_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				
				# Set macro_avg flag for this case
				macro_avg = False
		
		### Finetune Scheduler
		if freezing:
			for embedding in self.model.embeddings.embeddings:
				embedding.fine_tune = False
		dev_score_history = []
		dev_loss_history = []
		test_score_history = []
		test_loss_history = []
		train_loss_history = []
		# DataParallel disabled: codebase accesses .embeddings/.tag_type/.evaluate() etc.
		# directly on self.model which breaks under DataParallel wrapping.
		# Use CUDA_VISIBLE_DEVICES=0 or =1 to pick which GPU to use.
		# if self.n_gpu > 1:
		# 	self.model = torch.nn.DataParallel(self.model)

		
		score_list=[]
		
		name_list=sorted([x.name for x in self.model.embeddings.embeddings])
		# for faster quit training, use larger anneal factor to quitting
		min_learning_rate = learning_rate/1000
		
		curriculum=[]
		if curriculum_file is not None:
			with open(curriculum_file) as f:
				curriculum = json.loads(f.read())
		# pdb.set_trace()
		# self.model.embeddings.to('cpu')
		self.model.embeddings=self.model.embeddings.to('cpu')
		with torch.no_grad():
			if macro_avg:
				self.gpu_friendly_assign_embedding([batch_loader]+dev_loaders+test_loaders)
			else:
				self.gpu_friendly_assign_embedding([batch_loader,dev_loader,test_loader])
		# pdb.set_trace()
		
		try:
			for episode in range(start_episode,max_episodes):
				best_score=0
				learning_rate = default_learning_rate
				# reinitialize the optimizer and scheduler in training

				if len(self.update_params_group)>0:
					optimizer: torch.optim.Optimizer = self.optimizer(
						[{"params":other_params,"lr":learning_rate*lr_rate},
						{"params":self.update_params_group,"lr":learning_rate*lr_rate},
						{"params":finetune_params}
						],
						lr=learning_rate, **kwargs
					)
				else:
					optimizer: torch.optim.Optimizer = self.optimizer(
						[{"params":other_params,"lr":learning_rate*lr_rate},
						{"params":finetune_params}
						],
						lr=learning_rate, **kwargs
					)
				if self.optimizer_state is not None:
					optimizer.load_state_dict(self.optimizer_state)

				if use_amp:
					self.model, optimizer = amp.initialize(
						self.model, optimizer, opt_level=amp_opt_level
					)

				if not fine_tune_mode:
					if self.model.tag_type in dependency_tasks:
						scheduler = ExponentialLR(optimizer, decay**(1/decay_steps))
					else:
						anneal_mode = "min" if train_with_dev else "max"
						scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
							optimizer,
							factor=anneal_factor,
							patience=patience,
							mode=anneal_mode,
							verbose=True,
						)
				else:
					### Finetune Scheduler
					t_total = len(batch_loader) // gradient_accumulation_steps * max_epochs
					if rootschedule:
						warmup_steps = len(batch_loader)
						scheduler = get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, fix_embedding_steps = warmup_steps)
					else:
						if use_warmup:
							warmup_steps = len(batch_loader)
						scheduler = get_linear_schedule_with_warmup(
							optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
						)

				if self.scheduler_state is not None:
					scheduler.load_state_dict(self.scheduler_state) 
				




				log.info(
					f"================================== Start episode {episode + 1} =================================="
				)
				if self.controller.model_structure is not None:
					log.info("#### Current Training Action Distributions ####")
					self.assign_embedding_masks(batch_loader,sample=True, first_episode= episode == 0)
					log.info("#### Current Dev Action Distributions ####")
					if macro_avg and 'dev_loaders' in locals():
						for dev_loader in dev_loaders: 
							self.assign_embedding_masks(dev_loader,sample=False, first_episode= episode == 0)
					elif not macro_avg and 'dev_loader' in locals():
						self.assign_embedding_masks(dev_loader,sample=False, first_episode= episode == 0)
					log.info("#### Current Test Action Distributions ####")
					if macro_avg and 'test_loaders' in locals():
						for test_loader in test_loaders: 
							self.assign_embedding_masks(test_loader,sample=False, first_episode= episode == 0)
					elif not macro_avg and 'test_loader' in locals():
						self.assign_embedding_masks(test_loader,sample=False, first_episode= episode == 0)
					_ng = getattr(self.model, 'num_groups', 1)
					for _ni, _name in enumerate(name_list):
						log.info(f"  Embedding [{_ni}] {_name}")
					print(name_list)
				else:
					state = self.model.get_state()
					action, log_prob = self.controller.sample(
						state,
						num_groups=self.num_groups,
						ensure_entity_coverage=getattr(self, 'ensure_entity_coverage', False),
					)
					if episode == 0 and not random_search:
						log_prob = torch.log(torch.sigmoid(self.controller.get_value()))
						action = torch.ones_like(action)
						self.controller.previous_selection = action

					if curriculum_file is None:
						curriculum.append(action.cpu().tolist())
					else:
						action = torch.Tensor(curriculum[episode]).type_as(action)
					_ng = getattr(self.model, 'num_groups', 1)
					log.info(f"Episode {episode+1} action (num_groups={_ng}):")
					for _ni, _name in enumerate(name_list):
						_grp = action[_ni*_ng : (_ni+1)*_ng].int().tolist()
						_kept = sum(_grp)
						log.info(f"  {_name:<50s} groups={_grp}  kept={_kept}/{_ng}")
					print(name_list)
					print(action)
					print(self.controller(None))
					self.model.selection=action
			
				previous_learning_rate = learning_rate
				training_order = None
				bad_epochs2=0
				for epoch in range(0 + self.epoch, max_epochs + self.epoch):
					log_line(log)

					# get new learning rate
					if self.model.use_crf:
						learning_rate = optimizer.param_groups[0]["lr"]
					else:
						for group in optimizer.param_groups:
							learning_rate = group["lr"]
					if freezing and epoch == 1+self.epoch and fine_tune_mode:
						for embedding in self.model.embeddings.embeddings:
							if 'flair' in embedding.__class__.__name__.lower():
								embedding.fine_tune = False
								continue
							embedding.fine_tune = True
					# reload last best model if annealing with restarts is enabled
					if (
						learning_rate != previous_learning_rate
						and anneal_with_restarts
						and (base_path / "best-model.pt").exists()
					):
						log.info("resetting to best model")
						self.model.load(base_path / "best-model.pt")

					previous_learning_rate = learning_rate

					# stop training if learning rate becomes too small
					if learning_rate < min_learning_rate and warmup_steps <= 0:
						log_line(log)
						log.info("learning rate too small - quitting training!")
						log_line(log)
						break
					if bad_epochs2>=max_epochs_without_improvement:
						log_line(log)
						log.info(str(bad_epochs2) + " epochs after improvement - quitting training!")
						log_line(log)
						break
					if shuffle:
						batch_loader.reshuffle()
					if true_reshuffle:
						
						batch_loader.true_reshuffle()
						
						batch_loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
						
					self.model.train()
					self.controller.train()
					# TODO: check teacher parameters fixed and with eval() mode

					train_loss: float = 0

					seen_batches = 0
					#total_number_of_batches = sum([len(loader) for loader in batch_loader])
					total_number_of_batches = len(batch_loader)

					modulo = max(1, int(total_number_of_batches / 10))

					# process mini-batches
					batch_time = 0
					total_sent=0
					
					
					for batch_no, student_input in enumerate(batch_loader):
						# for group in optimizer.param_groups:
						#   temp_lr = group["lr"]
						# log.info('lr: '+str(temp_lr))
						# print(self.language_weight.softmax(1))
						# print(self.biaffine.U)
						
						start_time = time.time()
						total_sent+=len(student_input)
						try:
							
							loss = self.model.forward_loss(student_input)

							if self.model.use_decoder_timer:
								decode_time=time.time() - self.model.time
							optimizer.zero_grad()
							if self.n_gpu>1:
								loss = loss.mean()  # mean() to average on multi-gpu parallel training
							# Backward
							if use_amp:
								with amp.scale_loss(loss, optimizer) as scaled_loss:
									scaled_loss.backward()
							else:
								loss.backward()
								pass
						except Exception:
							traceback.print_exc()
							pdb.set_trace()
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
						if len(self.update_params_group)>0:
							torch.nn.utils.clip_grad_norm_(self.update_params_group, 5.0)
						optimizer.step()
						if (fine_tune_mode or self.model.tag_type in dependency_tasks):
							scheduler.step()

						seen_batches += 1
						train_loss += loss.item()

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(student_input, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(student_input,'features'):
							del student_input.features
						batch_time += time.time() - start_time
						if batch_no % modulo == 0:
							# print less information
							# if self.model.use_decoder_timer:
							#   log.info(
							#       f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
							#       f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f} - decode_sents/sec: {total_sent / decode_time:.2f}"
							#   )
								
							# else:
							#   log.info(
							#       f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
							#       f"{train_loss / seen_batches:.8f} - samples/sec: {total_sent / batch_time:.2f}"
							#   )
							total_sent = 0
							batch_time = 0
							iteration = epoch * total_number_of_batches + batch_no
							# if not param_selection_mode:
							#   weight_extractor.extract_weights(
							#       self.model.state_dict(), iteration
							#   )
					train_loss /= seen_batches

					self.model.eval()

					log_line(log)
					log.info(
						f"EPISODE {episode+1}, EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate}"
					)

					if self.use_tensorboard:
						writer.add_scalar("train_loss", train_loss, epoch + 1)

					# anneal against train loss if training with dev, otherwise anneal against dev score
					current_score = train_loss

					# evaluate on train / dev / test split depending on training settings
					result_line: str = ""

					if log_train:
						train_eval_result, train_loss = self.model.evaluate(
							batch_loader,
							embeddings_storage_mode=embeddings_storage_mode,
						)
						result_line += f"\t{train_eval_result.log_line}"

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						store_embeddings(self.corpus.train, embeddings_storage_mode)
						if embeddings_storage_mode == "none" and hasattr(self.corpus.train,'features'):
							del self.corpus.train.features
					log.info(f"==================Evaluating development set==================") 
					if log_dev:
						if macro_avg:
							
							if type(self.corpus) is ListCorpus:
								result_dict={}
								loss_list=[]
								print_sent='\n'
								
								for index, loader in enumerate(dev_loaders):
									if len(loader) == 0:
										continue
									# log_line(log)
									# log.info('current corpus: '+self.corpus.targets[index])
									current_result, dev_loss = self.model.evaluate(
										loader,
										embeddings_storage_mode=embeddings_storage_mode,
									)
									result_dict[self.corpus.targets[index]]=current_result.main_score*100
									print_sent+=self.corpus.targets[index]+'\t'+f'{result_dict[self.corpus.targets[index]]:.2f}'+'\t'
									loss_list.append(dev_loss)
									# log.info(current_result.log_line)
									# log.info(current_result.detailed_results)
							else:
								assert 0, 'not defined!'
							mavg=sum(result_dict.values())/len(result_dict)
							log.info('Macro Average: '+f'{mavg:.2f}'+'\tMacro avg loss: ' + f'{((sum(loss_list)/len(loss_list)).item()):.2f}' +  print_sent)
							dev_score_history.append(mavg)
							dev_loss_history.append((sum(loss_list)/len(loss_list)).item())
							
							current_score = mavg
						else:
							dev_eval_result, dev_loss = self.model.evaluate(
								dev_loader,
								embeddings_storage_mode=embeddings_storage_mode,
							)
							result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
							log.info(
								f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
							)
							# calculate scores using dev data if available
							# append dev score to score history
							dev_score_history.append(dev_eval_result.main_score)
							dev_loss_history.append(dev_loss)

							current_score = dev_eval_result.main_score
						
						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						if macro_avg:
							# For macro_avg, we need to handle multiple corpora
							if hasattr(self.corpus, 'dev_list'):
								for dev_corpus in self.corpus.dev_list:
									store_embeddings(dev_corpus, embeddings_storage_mode)
									if embeddings_storage_mode == "none" and hasattr(dev_corpus,'features'):
										del dev_corpus.features
						else:
							# Single corpus case
							store_embeddings(self.corpus.dev, embeddings_storage_mode)
							if embeddings_storage_mode == "none" and hasattr(self.corpus.dev,'features'):
								del self.corpus.dev.features
						
						if self.use_tensorboard:
							if macro_avg:
								writer.add_scalar("dev_loss", (sum(loss_list)/len(loss_list)).item(), epoch + 1)
								writer.add_scalar("dev_score", mavg, epoch + 1)
							else:
								writer.add_scalar("dev_loss", dev_loss, epoch + 1)
								writer.add_scalar("dev_score", dev_eval_result.main_score, epoch + 1)


					if current_score>=baseline_score or log_test:
						log.info(f"==================Evaluating test set==================")
						if macro_avg:
							
							if type(self.corpus) is ListCorpus:
								result_dict={}
								loss_list=[]
								print_sent='\n'
								for index, loader in enumerate(test_loaders):
									# log_line(log)
									# log.info('current corpus: '+self.corpus.targets[index])
									if len(loader) == 0:
										continue
									current_result, test_loss = self.model.evaluate(
										loader,
										embeddings_storage_mode=embeddings_storage_mode,
									)
									result_dict[self.corpus.targets[index]]=current_result.main_score*100
									print_sent+=self.corpus.targets[index]+'\t'+f'{result_dict[self.corpus.targets[index]]:.2f}'+'\t'
									loss_list.append(test_loss)
									# log.info(current_result.log_line)
									# log.info(current_result.detailed_results)
							else:
								assert 0, 'not defined!'
							mavg=sum(result_dict.values())/len(result_dict)
							log.info('Test Average: '+f'{mavg:.2f}'+'\tTest avg loss: ' + f'{((sum(loss_list)/len(loss_list)).item()):.2f}' +  print_sent)
							test_score_history.append(mavg)
							test_loss_history.append((sum(loss_list)/len(loss_list)).item())
							# When monitor_test is enabled, use test score for patience/best model
							if log_test:
								current_score = mavg
						else:
							test_eval_result, test_loss = self.model.evaluate(
								test_loader,
								embeddings_storage_mode=embeddings_storage_mode,
							)
							result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
							log.info(
								f"test : loss {test_loss} - score {test_eval_result.main_score}"
							)
							# calculate scores using test data if available
							# append test score to score history
							test_score_history.append(test_eval_result.main_score)
							test_loss_history.append(test_loss)
							# When monitor_test is enabled, use test score for patience/best model
							if log_test:
								current_score = test_eval_result.main_score

						# depending on memory mode, embeddings are moved to CPU, GPU or deleted
						if macro_avg:
							# For macro_avg, we need to handle multiple corpora
							if hasattr(self.corpus, 'test_list'):
								for test_corpus in self.corpus.test_list:
									store_embeddings(test_corpus, embeddings_storage_mode)
									if embeddings_storage_mode == "none" and hasattr(test_corpus,'features'):
										del test_corpus.features
						else:
							# Single corpus case
							store_embeddings(self.corpus.test, embeddings_storage_mode)
							if embeddings_storage_mode == "none" and hasattr(self.corpus.test,'features'):
								del self.corpus.test.features

						if self.use_tensorboard:
							if macro_avg:
								writer.add_scalar("test_loss", (sum(loss_list)/len(loss_list)).item(), epoch + 1)
								writer.add_scalar("test_score", mavg, epoch + 1)
							else:
								writer.add_scalar("test_loss", test_loss, epoch + 1)
								writer.add_scalar("test_score", test_eval_result.main_score, epoch + 1)


					# determine learning rate annealing through scheduler
					if not fine_tune_mode and self.model.tag_type not in dependency_tasks:
						scheduler.step(current_score)
					if current_score>best_score:
						best_score=current_score
						bad_epochs2=0
					else:
						bad_epochs2+=1
					train_loss_history.append(train_loss)

					# determine bad epoch number
					try:
						bad_epochs = scheduler.num_bad_epochs
					except:
						bad_epochs = 0
					for group in optimizer.param_groups:
						new_learning_rate = group["lr"]
					if new_learning_rate != previous_learning_rate:
						bad_epochs = patience + 1

					# log bad epochs
					log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")
					log.info(f"GLOBAL BAD EPOCHS (no improvement): {bad_epochs2}")

					# if checkpoint is enable, save model at each epoch
					# if checkpoint and not param_selection_mode:
					#   if self.n_gpu>1:
					#       self.model.module.save_checkpoint(
					#           base_path / "checkpoint.pt",
					#           optimizer.state_dict(),
					#           scheduler.state_dict(),
					#           epoch + 1,
					#           train_loss,
					#       )
					#   else:
					#       self.model.save_checkpoint(
					#           base_path / "checkpoint.pt",
					#           optimizer.state_dict(),
					#           scheduler.state_dict(),
					#           epoch + 1,
					#           train_loss,
					#       )

					# if we use dev data, remember best model based on dev evaluation score
					if (
						not train_with_dev
						and not param_selection_mode
						and current_score >= baseline_score
					):
						log.info(f"==================Saving the current overall best model: {current_score}==================") 
						gc.collect()
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
						if self.n_gpu>1:
							self.model.module.save(base_path / "best-model.pt")
						else:
							self.model.save(base_path / "best-model.pt")
							self.controller.save(base_path/ "controller.pt")
						baseline_score = current_score
						

				# if we do not use dev data for model selection, save final model
				# if save_final_model and not param_selection_mode:
				#   self.model.save(base_path / "final-model.pt")

				# pdb.set_trace()               
				log.info(
					f"================================== End episode {episode + 1} =================================="
				)
				# avoid back-propagation at the first iteration because reward is too large
				controller_optimizer.zero_grad()
				self.controller.zero_grad()
				if self.controller.model_structure is not None:
					if episode == 0:
						previous_best_score = best_score
						log.info(f"Setting baseline score to: {baseline_score}")
					else:
						base_reward = best_score - previous_best_score

						controller_loss = 0
						total_sent = 0
						if log_reward:
							base_reward = np.sign(base_reward)*np.log(np.abs(base_reward)+1)
						if sqrt_reward:
							base_reward = np.sign(base_reward)*np.sqrt(np.abs(base_reward))
						# pdb.set_trace()
						total_reward_at_each_position = torch.zeros(self.controller.num_actions).float().to(flair.device)
						for batch in batch_loader:

							action_change=torch.abs(batch.embedding_mask.to(flair.device)-batch.previous_embedding_mask.to(flair.device))
							reward = base_reward * (discount ** (action_change.sum(-1)-1))
							reward_at_each_position=reward[:,None]*action_change
							controller_loss+=-(batch.log_prob.to(flair.device)*reward_at_each_position).sum()
							total_sent+=len(batch)
							total_reward_at_each_position+=reward_at_each_position.sum(0)
						log.info(f"Current Reward at each position: {total_reward_at_each_position}")
						controller_loss/=total_sent
						controller_loss.backward()
						controller_optimizer.step()
					if best_score >= baseline_score:
						baseline_score = best_score
				else:
					if episode == 0:
						baseline_score = best_score
						log.info(f"Setting baseline score to: {baseline_score}")

						self.best_action = action
						self.controller.best_action = action
						_ng = getattr(self.model, 'num_groups', 1)
						log.info(f"Setting baseline action (num_groups={_ng}):")
						for _ni, _name in enumerate(name_list):
							_grp = self.best_action[_ni*_ng : (_ni+1)*_ng].int().tolist()
							log.info(f"  {_name:<50s} groups={_grp}  kept={sum(_grp)}/{_ng}")
					else:
						log.info(f"previous distributions: ")
						print(self.controller(None))
						# reward = best_score-baseline_score
						controller_loss = 0
						# pdb.set_trace()
						action_count = 0 
						average_reward = 0
						reward_at_each_position = torch.zeros_like(action)
						count_at_each_position = torch.zeros_like(action)
						if old_reward:
							# pdb.set_trace()
							reward = best_score - baseline_score
							reward_at_each_position += reward
						else:
							for prev_action in self.action_dict:
								reward = best_score - max(self.action_dict[prev_action]['scores'])
								prev_action = torch.Tensor(prev_action).type_as(action)
								if log_reward:
									reward = np.sign(reward)*np.log(np.abs(reward)+1)
								if sqrt_reward:
									reward = np.sign(reward)*np.sqrt(np.abs(reward))

								# reward* (discount^hamming_distance) to reduce the affect of long distance embeddings
								reward = reward * (discount ** (torch.abs(action-prev_action).sum()-1))
								average_reward += reward
								reward_at_each_position+=reward*torch.abs(action-prev_action)
								count_at_each_position+=torch.abs(action-prev_action)
								# controller_loss-=(log_prob*reward*torch.abs(action-prev_action)).sum()
								# remove the same action in the action_dict, since no reward
								if torch.abs(action-prev_action).sum() > 0:
									action_count+=1
						# controller_loss=controller_loss/action_count
						# pdb.set_trace()
						count_at_each_position[torch.where(count_at_each_position==0)]+=1
						controller_loss-=(log_prob*reward_at_each_position).sum()
						# controller_loss-=(log_prob*reward_at_each_position/count_at_each_position).sum()
						# only update the probability of embeddings that changes the selection compared to previous action
						# pdb.set_trace()
						# controller_loss = -(log_prob*reward*torch.abs(action-self.best_action)).sum()
						if random_search:
							log.info('================= Doing random search, stop updating the controller =================')
						else:
							controller_loss.backward()
							print("#=================")
							print(self.controller.selector)
							print(self.controller.selector.grad)
							# print(self.controller.selector - self.controller.selector.grad*self.controller_learning_rate)
							# pdb.set_trace()
							controller_optimizer.step()
							print(self.controller.selector)
							print("#=================")
							# pdb.set_trace()
						
						log.info(f"After distributions: ")
						print(self.controller(None))
						# pdb.set_trace()
						if best_score >= baseline_score:
							baseline_score = best_score
							self.best_action = action
							self.controller.best_action = action
							log.info(f"Setting baseline score to: {baseline_score}")
							_ng = getattr(self.model, 'num_groups', 1)
							log.info(f"Setting baseline action (num_groups={_ng}):")
							for _ni, _name in enumerate(name_list):
								_grp = self.best_action[_ni*_ng : (_ni+1)*_ng].int().tolist()
								log.info(f"  {_name:<50s} groups={_grp}  kept={sum(_grp)}/{_ng}")

						log.info('=============================================')  
						_ng = getattr(self.model, 'num_groups', 1)
						log.info(f"Current Action (num_groups={_ng}):")
						for _ni, _name in enumerate(name_list):
							_grp = action[_ni*_ng : (_ni+1)*_ng].int().tolist()
							log.info(f"  {_name:<50s} groups={_grp}  kept={sum(_grp)}/{_ng}")
						log.info(f"Overall best score: {baseline_score}")
						log.info(f"State dictionary: {self.action_dict}")
						log.info('=============================================')
						
					# pdb.set_trace()
					curr_action = tuple(action.cpu().tolist())
					if curr_action not in self.action_dict:
						self.action_dict[curr_action] = {}
						self.action_dict[curr_action]['counts']=0
						self.action_dict[curr_action]['scores']=[]
						# self.action_dict[curr_action]['scores'].append(best_score)
					self.action_dict[curr_action]['counts']+=1
					self.action_dict[curr_action]['scores'].append(best_score)

				training_state = {
								'episode':episode,
								'best_action':self.best_action if self.controller.model_structure is None else None,
								'baseline_score': baseline_score,
								'action_dict': self.action_dict,
								'num_groups': self.num_groups,
								'group_perm_indices': getattr(self.model, 'group_perm_indices', None),
								'group_masks': getattr(self.model, 'group_masks', None),
								}
				torch.save(training_state,base_path/'training_state.pt')
				torch.save(controller_optimizer.state_dict(),base_path/'controller_optimizer_state.pt')
				# pdb.set_trace()

		except KeyboardInterrupt:
			log_line(log)
			log.info("Exiting from training early.")

			if self.use_tensorboard:
				writer.close()

			if not param_selection_mode:
				log.info("Saving model ...")
				self.model.save(base_path / "final-model.pt")
				log.info("Done.")
		# pdb.set_trace()
		if self.controller.model_structure is None:
			_ng = getattr(self.model, 'num_groups', 1)
			_final_action = self.controller(state) >= 0.5
			log.info(f"Final controller probabilities (num_groups={_ng}):")
			for _ni, _name in enumerate(name_list):
				_grp = _final_action[_ni*_ng : (_ni+1)*_ng].int().tolist()
				log.info(f"  {_name:<50s} groups={_grp}  kept={sum(_grp)}/{_ng}")
			print(name_list)
			print(_final_action)

			for action in self.action_dict:
				self.action_dict[action]['average']=sum(self.action_dict[action]['scores'])/self.action_dict[action]['counts']
			log.info(f"Final State dictionary: {self.action_dict}")
			self.model.selection=self.best_action
			with open(base_path/"curriculum.json",'w') as f:
				f.write(json.dumps(curriculum))

		# test best model if test data is present
		if self.corpus.test:
			final_score = self.final_test(base_path, eval_mini_batch_size, num_workers)
		else:
			final_score = 0
			log.info("Test data not provided setting final score to 0")
		log.removeHandler(log_handler)
		if self.use_tensorboard:
			writer.close()
		if self.model.use_language_attention:
			if self.model.biaf_attention:
				print(language_weight.softmax(1))
			else:
				print(self.language_weight.softmax(1))
		return {
			"test_score": final_score,
			"dev_score_history": dev_score_history,
			"test_score_history": test_score_history,
			"train_loss_history": train_loss_history,
			"dev_loss_history": dev_loss_history,
			"test_loss_history": test_loss_history,
		}
	@property
	def interpolation(self):
		try:
			return self.config['interpolation']
		except:
			return 0.5
	@property
	def teacher_annealing(self):
		try:
			return self.config['teacher_annealing']
		except:
			return False
	@property
	def anneal_factor(self):
		try:
			return self.config['anneal_factor']
		except:
			return 2

	def assign_embedding_masks(self, data_loader, sample=False, first_episode=False):
		lang_dict = {}
		distr_dict = {}
		for batch_no, sentences in enumerate(data_loader):
			
			lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
			longest_token_sequence_in_batch: int = max(lengths)
			
			self.model.embeddings.embed(sentences)
			sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())],-1)
			mask=self.model.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).to(flair.device).type_as(sentence_tensor)

			# sum over all embeddings to get the sentence level features
			# sentence_feature = (sentence_tensor*mask).sum(-2)/mask.sum(-1)

			# given the sentence feature, calculate the embedding mask selection and log probability
			sentence_tensor = sentence_tensor.detach()
			if sample:
				selection, log_prob = self.controller.sample(sentence_tensor,mask)
				# pdb.set_trace()
				selection = selection.to('cpu')
				log_prob = log_prob.to('cpu')
				sentences.log_prob = log_prob
			else:
				prediction = self.controller(sentence_tensor,mask)
				selection = prediction >= 0.5
				for idx in range(len(selection)):
					if selection[idx].sum() == 0:
						# pdb.set_trace()
						selection[idx][torch.argmax(prediction[idx])]=1
						# m_temp = torch.distributions.Bernoulli(one_prob[idx])
						# selection[idx] = m_temp.sample()

				selection = selection.to('cpu')
			
			if first_episode:
				selection = torch.ones_like(selection)

			# for idx in range(len(selection)):
			#   if sentences[idx].lang_id == 0:
			#       selection[idx,0]=1
			#       selection[idx,1]=0
			#   if sentences[idx].lang_id == 1:
			#       selection[idx,0]=0
			#       selection[idx,1]=1

			if hasattr(sentences,'embedding_mask'):
				sentences.previous_embedding_mask = sentences.embedding_mask
			sentences.embedding_mask = selection
			# pdb.set_trace()
			distribution=self.controller(sentence_tensor,mask)
			for sent_id, sentence in enumerate(sentences):
				if hasattr(sentence,'embedding_mask'):
					sentence.previous_embedding_mask = selection[sent_id]
				sentence.embedding_mask = selection[sent_id]
				if sample:
					sentence.log_prob = log_prob[sent_id]

				if sentence.lang_id not in lang_dict:
					lang_dict[sentence.lang_id] = []
					distr_dict[sentence.lang_id] = []
				lang_dict[sentence.lang_id].append(selection[sent_id])
				distr_dict[sentence.lang_id].append(distribution[sent_id])
			

		# pdb.set_trace()
		for lang_id in lang_dict:
			print(self.id2corpus[lang_id], (sum(lang_dict[lang_id])/len(lang_dict[lang_id])).tolist())
			print(self.id2corpus[lang_id], (sum(distr_dict[lang_id])/len(distr_dict[lang_id])).tolist())
		return


	# def gpu_friendly_assign_embedding(self,loaders):
	#   # pdb.set_trace()
	#   for embedding in self.model.embeddings.embeddings:
	#       if ('WordEmbeddings' not in embedding.__class__.__name__ and 'Char' not in embedding.__class__.__name__ and 'Lemma' not in embedding.__class__.__name__ and 'POS' not in embedding.__class__.__name__) and not (hasattr(embedding,'fine_tune') and embedding.fine_tune):
	#           print(embedding.name, count_parameters(embedding))
	#           # 
	#           embedding.to(flair.device)
	#           for loader in loaders:
	#               for sentences in loader:
	#                   lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
	#                   longest_token_sequence_in_batch: int = max(lengths)
	#                   # if longest_token_sequence_in_batch>100:
	#                   #   pdb.set_trace()
	#                   embedding.embed(sentences)
	#                   store_embeddings(sentences, self.embeddings_storage_mode)
	#           embedding=embedding.to('cpu')
	#       else:
	#           embedding=embedding.to(flair.device)
	#   # torch.cuda.empty_cache()
	#   log.info("Finished Embeddings Assignments")
	#   return 
	# def assign_predicted_embeddings(self,doc_dict,embedding,file_name):
	#   # torch.cuda.empty_cache()
	#   lm_file = h5py.File(file_name, "r")
	#   for key in doc_dict:
	#       if key == 'start':
	#           for i, sentence in enumerate(doc_dict[key]):
	#               for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
	#                   word_embedding = torch.zeros(embedding.embedding_length).float()
	#                   word_embedding = torch.FloatTensor(word_embedding)

	#                   token.set_embedding(embedding.name, word_embedding)
	#           continue
	#       group = lm_file[key]
	#       num_sentences = len(list(group.keys()))
	#       sentences_emb = [group[str(i)][...] for i in range(num_sentences)]
	#       try: 
	#           assert len(doc_dict[key])==len(sentences_emb)
	#       except:
	#           pdb.set_trace()
	#       for i, sentence in enumerate(doc_dict[key]):
	#           for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
	#               word_embedding = sentences_emb[i][token_idx]
	#               word_embedding = torch.from_numpy(word_embedding).view(-1)

	#               token.set_embedding(embedding.name, word_embedding)
	#           store_embeddings([sentence], 'cpu')
	#       # for idx, sentence in enumerate(doc_dict[key]):
	#   log.info("Loaded predicted embeddings: "+file_name)
	#   return 
	
	def resort(self,loader,is_crf=False, is_posterior=False, is_token_att=False):
		for batch in loader:
			if is_posterior:
				try:
					posteriors=[x._teacher_posteriors for x in batch]
					posterior_lens=[len(x[0]) for x in posteriors]
					lens=posterior_lens.copy()
					targets=posteriors.copy()
				except:
					pdb.set_trace()
			if is_token_att:
				sentfeats=[x._teacher_sentfeats for x in batch]
				sentfeats_lens=[len(x[0]) for x in sentfeats]
			#     lens=sentfeats_lens.copy()
			#     targets=sentfeats.copy()
			if is_crf:
				targets=[x._teacher_target for x in batch]
				lens=[len(x[0]) for x in targets]
				if hasattr(self.model,'distill_rel') and self.model.distill_rel:
					rel_targets=[x._teacher_rel_target for x in batch]
			if not is_crf and not is_posterior:
				targets=[x._teacher_prediction for x in batch]
				if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
					rel_targets=[x._teacher_rel_prediction for x in batch]
				lens=[len(x[0]) for x in targets]
			sent_lens=[len(x) for x in batch]

			if is_posterior:
				assert posterior_lens==lens, 'lengths of two targets not match'
			
			if max(lens)>min(lens) or max(sent_lens)!=max(lens) or (is_posterior and self.model.tag_type=='dependency'):
				# if max(sent_lens)!=max(lens):
				max_shape=max(sent_lens)
				for index, target in enumerate(targets):
					new_targets=[]
					new_rel_targets=[]
					new_posteriors=[]
					new_sentfeats=[]
					if is_posterior:
						post_vals=posteriors[index]
					if is_token_att:
						sentfeats_vals=sentfeats[index]
					for idx, val in enumerate(target):
						if self.model.tag_type=='dependency':
							if is_crf:
								shape=[max_shape]+list(val.shape[1:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index]]=val[:sent_lens[index]]
								new_targets.append(new_target)
								if hasattr(self.model,'distill_rel') and self.model.distill_rel:
									cur_val = rel_targets[index][idx]
									rel_shape=[max_shape]+list(cur_val.shape[1:])
									new_rel_target=torch.zeros(rel_shape).type_as(cur_val)
									new_rel_target[:sent_lens[index]]=cur_val[:sent_lens[index]]
									new_rel_targets.append(new_rel_target)
							if not is_crf and not is_posterior:
								shape=[max_shape,max_shape]+list(val.shape[2:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index],:sent_lens[index]]=val[:sent_lens[index],:sent_lens[index]]
								new_targets.append(new_target)
								if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
									cur_val = rel_targets[index][idx]
									rel_shape=[max_shape,max_shape]+list(cur_val.shape[2:])
									new_rel_target=torch.zeros(rel_shape).type_as(cur_val)
									new_rel_target[:sent_lens[index],:sent_lens[index]]=cur_val[:sent_lens[index],:sent_lens[index]]
									new_rel_targets.append(new_rel_target)
							if is_posterior:
								post_val=post_vals[idx]
								# shape=[max_shape-1,max_shape-1] + list(post_val.shape[2:])
								shape=[max_shape,max_shape] + list(post_val.shape[2:])
								# if max_shape==8:
								#   pdb.set_trace()
								new_posterior=torch.zeros(shape).type_as(post_val)
								# remove the root token
								# new_posterior[:sent_lens[index]-1,:sent_lens[index]-1]=post_val[:sent_lens[index]-1,:sent_lens[index]-1]
								new_posterior[:sent_lens[index],:sent_lens[index]]=post_val[:sent_lens[index],:sent_lens[index]]
								new_posteriors.append(new_posterior)
						else:
							if is_crf or (not is_crf and not is_posterior):
								shape=[max_shape]+list(val.shape[1:])+list(val.shape[2:])
								new_target=torch.zeros(shape).type_as(val)
								new_target[:sent_lens[index]]=val[:sent_lens[index]]
								new_targets.append(new_target)
							if is_token_att:
								sentfeats_val=sentfeats_vals[idx]
								shape=[max_shape]+list(sentfeats_val.shape[1:])
								new_sentfeat=torch.zeros(shape).type_as(sentfeats_val)
								new_sentfeat[:sent_lens[index]]=sentfeats_val[:sent_lens[index]]
								new_sentfeats.append(new_sentfeat)
							if is_posterior:
								post_val=post_vals[idx]
								shape=[max_shape]+list(post_val.shape[1:])
								new_posterior=torch.zeros(shape).type_as(post_val)
								new_posterior[:sent_lens[index]]=post_val[:sent_lens[index]]
								new_posteriors.append(new_posterior)
							
					if is_crf:
						batch[index]._teacher_target=new_targets
						if hasattr(self.model,'distill_rel') and  self.model.distill_rel:
							batch[index]._teacher_rel_target=new_rel_targets
					if is_posterior:
						batch[index]._teacher_posteriors=new_posteriors
					if is_token_att:
						batch[index]._teacher_sentfeats=new_sentfeats
					if not is_crf and not is_posterior:
						if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
							batch[index]._teacher_rel_prediction=new_rel_targets
						batch[index]._teacher_prediction=new_targets
			if hasattr(batch,'teacher_features'):
				if is_posterior:
					batch.teacher_features['posteriors']=torch.stack([sentence.get_teacher_posteriors() for sentence in batch],0).cpu()
					# lens=[len(x) for x in batch]
					# posteriors = batch.teacher_features['posteriors']
					# if max(lens) == posteriors.shape[-1]:
					#   pdb.set_trace()
				if (not is_crf and not is_posterior):
					batch.teacher_features['distributions'] = torch.stack([sentence.get_teacher_prediction() for sentence in batch],0).cpu()
					if hasattr(self.model, 'distill_factorize') and self.model.distill_factorize:
						batch.teacher_features['rel_distributions'] = torch.stack([sentence.get_teacher_rel_prediction() for sentence in batch],0).cpu()
				if is_crf:
					batch.teacher_features['topk']=torch.stack([sentence.get_teacher_target() for sentence in batch],0).cpu()
					if self.model.crf_attention or self.model.tag_type=='dependency':
						batch.teacher_features['weights']=torch.stack([sentence.get_teacher_weights() for sentence in batch],0).cpu()
					if hasattr(self.model,'distill_rel') and self.model.distill_rel:
						batch.teacher_features['topk_rels']=torch.stack([sentence.get_teacher_rel_target() for sentence in batch],0).cpu()
		return loader
	def assign_corpus(self, corpus, set_name = 'test_', corpus_name = 'CONLL_03_ENGLISH', train_with_doc = True, pretrained_file_dict: dict = {}
	):	
		doc_sentence_dict = {}
		doc_sentence_dict = self.assign_documents(corpus, set_name, doc_sentence_dict, corpus_name, train_with_doc)
		new_sentences=[]
		for sentid, sentence in enumerate(corpus):
			if sentence[0].text=='-DOCSTART-':
				continue
			new_sentences.append(sentence)
		corpus.sentences = new_sentences.copy()
		corpus.reset_sentence_count
			
		for embedding in self.model.embeddings.embeddings:
			if embedding.name in pretrained_file_dict:
				self.assign_predicted_embeddings(doc_sentence_dict,embedding,pretrained_file_dict[embedding.name])
		return corpus

	def final_test(
		self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8, overall_test: bool = True, quiet_mode: bool = False, nocrf: bool = False, predict_posterior: bool = False, debug: bool = False, keep_embedding: int = -1, sort_data=False, mst = False
	):

		log_line(log)
		

		self.model.eval()
		self.model.to('cpu')
		name_list=sorted([x.name for x in self.model.embeddings.embeddings])
		if quiet_mode:
			#blockPrint()
			log.disabled=True
		# pdb.set_trace()
		if (base_path / "best-model.pt").exists():
			self.model = self.model.load(base_path / "best-model.pt", device='cpu')
			log.info("Testing using best model ...")
		elif (base_path / "final-model.pt").exists():
			self.model = self.model.load(base_path / "final-model.pt", device='cpu')
			log.info("Testing using final model ...")
		try:
			if self.controller.model_structure is not None:
				self.controller = self.controller.load(base_path / "controller.pt")
				log.info("Testing using best controller ...")
			if self.controller.model_structure is None:
				training_state = torch.load(base_path/'training_state.pt')
				self.best_action = training_state.get('best_action', None)
				self.model.selection=self.best_action
				# Restore Grouped-ACE / Entity-Golden attributes
				self.model.num_groups = training_state.get('num_groups', 1)
				self.model.group_perm_indices = training_state.get('group_perm_indices', getattr(self.model, 'group_perm_indices', None))
				self.model.group_masks = training_state.get('group_masks', None)
			
				_ng = getattr(self.model, 'num_groups', 1)
				log.info(f"Setting embedding mask to the best action (num_groups={_ng}):")
				for _ni, _name in enumerate(name_list):
					_grp = self.best_action[_ni*_ng : (_ni+1)*_ng].int().tolist()
					log.info(f"  {_name:<50s} groups={_grp}  kept={sum(_grp)}/{_ng}")
				print(name_list)
		except:
			pdb.set_trace()

		# Since there are a lot of embeddings, we keep these embeddings to cpu in order to avoid OOM
		for name, module in self.model.named_modules():
			if 'embeddings' in name or name == '':
				continue
			else:
				module.to(flair.device)
		parameters = [x for x in self.model.named_parameters()]
		for parameter in parameters:
			name = parameter[0]
			module = parameter[1]
			module.data.to(flair.device)
			if '.' not in name:
				if type(getattr(self.model, name))==torch.nn.parameter.Parameter:
					setattr(self.model, name, torch.nn.parameter.Parameter(getattr(self.model,name).to(flair.device)))

		# if hasattr(self.model,'transitions'):
		#   self.model.transitions = torch.nn.parameter.Parameter(self.model.transitions.to(flair.device))
		if mst == True:
			self.model.is_mst=mst
		for embedding in self.model.embeddings.embeddings:
			embedding.to('cpu')
		if debug:
			self.model.debug=True
			# if hasattr(self.model,'transitions'):
			#   self.model.transitions = torch.nn.Parameter(torch.randn(self.model.tagset_size, self.model.tagset_size).to(flair.device))
		else:
			self.model.debug=False
		if nocrf:
			self.model.use_crf=False
		if predict_posterior:
			self.model.predict_posterior=True
		if keep_embedding>-1:
			self.model.keep_embedding=keep_embedding

		for embedding in self.model.embeddings.embeddings:
			# manually fix the bug for the tokenizer becoming None
			if hasattr(embedding,'tokenizer') and embedding.tokenizer is None:
				from transformers import AutoTokenizer
				name = embedding.name
				if hasattr(embedding,'model_path'):
					name=embedding.model_path
				if '_v2doc' in name:
					name = name.replace('_v2doc','')
				if '_extdoc' in name:
					name = name.replace('_extdoc','')
				try:
					embedding.tokenizer = AutoTokenizer.from_pretrained(name)
				except:
					temp_name = name.split('/')[-1]
					# temp_name = './'+temp_name
					embedding.tokenizer = AutoTokenizer.from_pretrained(temp_name)
			if hasattr(embedding,'model') and hasattr(embedding.model,'encoder') and not hasattr(embedding.model.encoder,'config'):
				embedding.model.encoder.config = embedding.model.config
		# pdb.set_trace()
		if overall_test:
			loader=ColumnDataLoader(list(self.corpus.test),eval_mini_batch_size, use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
			loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
			with torch.no_grad():
				self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
				if self.controller.model_structure is not None:
					self.assign_embedding_masks(loader,sample=False)
			test_results, test_loss = self.model.evaluate(
				loader,
				out_path=base_path / "test.tsv",
				embeddings_storage_mode="cpu",
				prediction_mode=True,
			)
			test_results: Result = test_results
			log.info(test_results.log_line)
			log.info(test_results.detailed_results)
			log_line(log)
		if quiet_mode:
			enablePrint()
			if overall_test:
				if keep_embedding>-1:
					embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
					embedding_name = '_'.join(embedding_name)
					if 'lm-' in embedding_name.lower():
						embedding_name = 'Flair'
					elif 'bert' in embedding_name.lower():
						embedding_name = 'MBERT'
					elif 'word' in embedding_name.lower():
						embedding_name = 'Word'
					elif 'char' in embedding_name.lower():
						embedding_name = 'char'
					print(embedding_name,end=' ')
				print('Average', end=' ')
				print(test_results.main_score, end=' ')

		# if we are training over multiple datasets, do evaluation for each
		if type(self.corpus) is MultiCorpus:
			for subcorpus in self.corpus.corpora:
				log_line(log)
				log.info('current corpus: '+subcorpus.name)
				loader=ColumnDataLoader(list(subcorpus.test),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				with torch.no_grad():
					self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
					if self.controller.model_structure is not None:
						self.assign_embedding_masks(loader,sample=False)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{subcorpus.name}-test.tsv",
					embeddings_storage_mode="none",
					prediction_mode=True,
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					if keep_embedding>-1:
						embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
						embedding_name = '_'.join(embedding_name)
						if 'lm-' in embedding_name.lower() or 'forward' in embedding_name.lower() or 'backward' in embedding_name.lower():
							embedding_name = 'Flair'
						elif 'bert' in embedding_name.lower():
							embedding_name = 'MBERT'
						elif 'word' in embedding_name.lower():
							embedding_name = 'Word'
						elif 'char' in embedding_name.lower():
							embedding_name = 'char'
						print(embedding_name,end=' ')
					print(subcorpus.name,end=' ')
					print(current_result.main_score,end=' ')

		elif type(self.corpus) is ListCorpus:
			for index,subcorpus in enumerate(self.corpus.test_list):
				log_line(log)
				log.info('current corpus: '+self.corpus.targets[index])
				loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=self.use_bert,tokenizer=self.bert_tokenizer, model = self.model, sentence_level_batch = self.sentence_level_batch, sort_data=sort_data)
				loader.assign_tags(self.model.tag_type,self.model.tag_dictionary)
				with torch.no_grad():
					self.gpu_friendly_assign_embedding([loader], selection = self.model.selection)
					if self.controller.model_structure is not None:
						self.assign_embedding_masks(loader,sample=False)
				current_result, test_loss = self.model.evaluate(
					loader,
					out_path=base_path / f"{self.corpus.targets[index]}-test.tsv",
					embeddings_storage_mode="none",
					prediction_mode=True,
				)
				log.info(current_result.log_line)
				log.info(current_result.detailed_results)
				if quiet_mode:
					if keep_embedding>-1:
						embedding_name = sorted(loader[0].features.keys())[keep_embedding].split()
						embedding_name = '_'.join(embedding_name)
						if 'lm-' in embedding_name.lower() or 'forward' in embedding_name.lower() or 'backward' in embedding_name.lower():
							embedding_name = 'Flair'
						elif 'bert' in embedding_name.lower():
							embedding_name = 'MBERT'
						elif 'word' in embedding_name.lower():
							embedding_name = 'Word'
						elif 'char' in embedding_name.lower():
							embedding_name = 'char'
						print(embedding_name,end=' ')
					print(self.corpus.targets[index],end=' ')
					print(current_result.main_score,end=' ')
		if keep_embedding<0:
			print()
		if overall_test:
			# get and return the final test score of best model
			final_score = test_results.main_score

			return final_score
		return 0

