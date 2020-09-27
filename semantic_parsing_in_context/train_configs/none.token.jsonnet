{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "sparc",
		"lazy": false,
		"loading_limit": -1,
		"context_mode": "none"
	},
	"model": {
		"type": "sparc",
		"loss_mask": 8,
		"serialization_dir": "",
		"text_embedder": {
			"tokens": {
				"type": "embedding",
				"embedding_dim": 100,
				"trainable": true,
				"padding_index": 0
			}
		},
		"action_embedding_dim": 100,
		"entity_embedding_dim": 100,
		"text_encoder": {
			"type": "lstm",
			"input_size": 200,
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
		"use_feature_score": true,
		"use_schema_encoder": true,
		"use_linking_embedding": true,
		"sql_hidden_size": 100,
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
			"lr": 1e-3
		},
		"num_serialized_models_to_keep": 10,
		"should_log_learning_rate": true
	}
}