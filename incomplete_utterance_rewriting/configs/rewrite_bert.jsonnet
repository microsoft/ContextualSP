{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "rewrite",
		"lazy": false,
		"super_mode": "before",
		"joint_encoding": true,
		"language": "zh",
		"use_bert": true,
		"extra_stop_words": ["的", "是", "我", "了", "和"]
	},
	"model": {
		"type": "rewrite",
		"word_embedder": {
			"bert": {
				"type": "bert-pretrained",
				"pretrained_model": "bert-base-chinese",
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
		"text_encoder": {
			"type": "lstm",
			"input_size": 768,
			"hidden_size": 200,
			"bidirectional": true,
			"num_layers": 1
		},
		"inp_drop_rate": 0.2,
		"out_drop_rate": 0.2,
		"feature_sel": 114,
		"loss_weights": [0.25, 0.4, 0.35],
		"super_mode": "before"
	},
	"iterator": {
		"type": "basic",
		"batch_size": 16
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 16
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+EM",
		"optimizer": {
			"type": "adam",
			"parameter_groups": [
				[
					[
						".*word_embedder.*"
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