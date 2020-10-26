{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "rewrite",
		"lazy": false,
		"super_mode": "before",
		"joint_encoding": true,
		"extra_stop_words": ["的", "是", "我", "了", "和"]
	},
	"model": {
		"type": "rewrite",
		"word_embedder": {
			"tokens": {
				"type": "embedding",
				"embedding_dim": 100,
				"trainable": true,
				"padding_index": 0
			}
		},
		"text_encoder": {
			"type": "lstm",
			"input_size": 100,
			"hidden_size": 200,
			"bidirectional": true,
			"num_layers": 1
		},
		"inp_drop_rate": 0.5,
		"out_drop_rate": 0.5,
		"feature_sel": 114,
		"loss_weights": [0.25, 0.4, 0.35],
		"super_mode": "before"
	},
	"iterator": {
		"type": "basic",
		"batch_size": 24
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 24
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+EM",
		"optimizer": {
			"type": "adam",
			"lr": 2e-3
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