{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "rewrite",
		"lazy": false,
		"super_mode": "before",
		"joint_encoding": true,
		"extra_stop_words": [
			"'s",
			"besides",
			"the",
			"in",
			"of"
		]
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
		"inp_drop_rate": 0.2,
		"out_drop_rate": 0.2,
		"feature_sel": 115,
		"loss_weights": [
			0.1,
			0.4,
			0.5
		],
		"super_mode": "before",
		"unet_down_channel": 128,
		"enable_training_log": false
	},
	"iterator": {
		"type": "basic",
		"batch_size": 4
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 4
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+BLEU4",
		"optimizer": {
			"type": "adam",
			"lr": 2e-4
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