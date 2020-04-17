{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "sparc",
		"lazy": false,
		"loading_limit": -1,
		"context_mode": "turn",
		"bert_mode": "v3",
		"utterance_token_indexers": {
			"bert": {
				"type": "bert-pretrained",
				"pretrained_model": "bert-base-uncased",
				"do_lowercase": true,
				"never_lowercase": [
					"[UNK]",
					"[SEP]",
					"[PAD]",
					"[CLS]",
					"[MASK]"
				],
				"use_starting_offsets": false,
				"truncate_long_sequences": false
			}
		}
	},
	"model": {
		"type": "sparc",
		"loss_mask": 8,
		"serialization_dir": "",
		"text_embedder": {
			"bert": {
				"type": "bert-pretrained",
				"pretrained_model": "bert-base-uncased",
				"top_layer_only": true,
				"requires_grad": true
			},
			"allow_unmatched_keys": true,
			"embedder_to_indexer_map": {
				"bert": [
					"bert",
					"bert-offsets",
					"bert-type-ids"
				]
			}
		},
		"action_embedding_dim": 100,
		"entity_embedding_dim": 768,
		"text_encoder": {
			"type": "lstm",
			"input_size": 868,
			"hidden_size": 200,
			"bidirectional": true,
			"num_layers": 1
		},
		"decoder_beam_search": {
			"beam_size": 10
		},
		"training_beam_size": 1,
		"max_decoding_steps": 100,
		"input_attention": {
			"type": "dot_product"
		},
		"bert_mode": "v3",
		"use_feature_score": true,
		"use_schema_encoder": true,
		"use_linking_embedding": true,
		"use_discourse_encoder": true,
		"use_attend_over_history": true,
		"use_turn_position": true,
		"use_copy_token": true,
		"copy_encode_with_context": true,
		"copy_encode_anon": false,
		"dropout_rate": 0.5
	},
	"iterator": {
		"type": "basic",
		"batch_size": 8
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 1
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+sql_exact_match",
		"optimizer": {
			"type": "adam",
			"parameter_groups": [
				[
					[
						".*text_embedder.*"
					],
					{
						"lr": 1e-5
					}
				]
			],
			"lr": 1e-3
		},
		"learning_rate_scheduler": {
			"type": "reduce_on_plateau",
			"factor": 0.5,
			"mode": "max",
			"patience": 5
		},
		"num_serialized_models_to_keep": 10,
		"should_log_learning_rate": true
	}
}