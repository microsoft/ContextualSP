from os.path import join

from tqdm import tqdm

from inference.bind_types import *
from inference.inference_models import *
from inference.pipeline_base import *
from models import *
from utils import *
from inference import *
from contracts import *
from transformers import PreTrainedTokenizer
from inference.pipeline_torchscript import NLBindingTorchScriptPipeline
import torch

infer_input_names = ['input_token_ids', 'entity_indices', 'question_indices']


# 读取文件的每一行, 返回列表
def get_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        for s in f.readlines():
            s = s.strip()
            if s:
                lines.append(s)
        return lines


def load_json_file(filename):
    """
    :param filename: 文件名
    :return: 数据对象，json/list
    """
    with open(filename, encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_file(filename):
    """
    :param filename: 文件名
    :return: 数据对象，json/list
    """
    data = []
    for line in get_lines(filename):
        js = json.loads(line)
        data.append(js)
    return data


def save_json_file(filename, data, indent=2):
    """
    :param filename: 输出文件名
    :param data: 数据对象，json/list
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as fp:
        if indent:
            json.dump(data, fp, indent=indent, ensure_ascii=False)
        else:
            json.dump(data, fp, ensure_ascii=False)
    print('save file %s successful!' % filename)


def convert_to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.cpu().numpy()

    if isinstance(inputs, list):
        return [convert_to_numpy(x) for x in inputs]

    if isinstance(inputs, dict):
        return {key: convert_to_numpy(val) for key, val in inputs.items()}

    raise NotImplementedError(inputs)


def convert_to_request(example: Text2SQLExample) -> NLBindingRequest:
    return NLBindingRequest(
        language=example.language,
        question_tokens=[NLToken.from_json(x.to_json()) for x in example.question.tokens],
        columns=[NLColumn.from_json(x.to_json()) for x in example.schema.columns],
        matched_values=[
            NLMatchedValue(column=x.column, name=x.name, tokens=[NLToken.from_json(xx.to_json()) for xx in x.tokens],
                           start=x.start, end=x.end) for x in example.matched_values]
    )


def load_data_inputs(path: str, tokenizer: PreTrainedTokenizer, sampling_size: int = -1,
                     infer_input_names: List[str] = infer_input_names) -> List[Dict]:
    examples: List[Text2SQLExample] = load_json_objects(Text2SQLExample, path)
    examples = [ex.ignore_unmatched_values() for ex in examples if ex.value_resolved]

    if sampling_size > 0:
        print()
        print('random sampling {} examples ...'.format(sampling_size))
        examples = examples[:sampling_size]  # random.sample(examples, sampling_size)

    data_iter = load_data_loader(
        examples=examples,
        tokenizer=tokenizer,
        batch_size=1,
        is_training=False,
        max_enc_length=tokenizer.model_max_length,
        n_processes=1)

    data_inputs, data_requests, cp_labels = [], [], []
    for batch_inputs in data_iter:
        data_inputs += [[batch_inputs[key][0] for key in infer_input_names]]

        data_requests += [convert_to_request(batch_inputs['example'][0])]

        cp_label = batch_inputs['concept_labels'][0].cpu().tolist()
        cp_labels += [cp_label]

    return data_inputs, data_requests, cp_labels


def load_model_as_torchscript(ckpt_path: str, dummy_input) -> BaseGroundingModel:
    config_path = os.path.join(os.path.dirname(ckpt_path), 'model_config.json')
    print("Load model config from {} ...".format(config_path))
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    model = UniGroundingModel(config=config, torch_script=True)

    traced_cpu = torch.jit.trace(model.infer, dummy_input)
    cpu_script_path = ckpt_path.replace(".pt", ".script_cpu.bin")
    torch.jit.save(traced_cpu, cpu_script_path)
    print("Dump scripted cpu model into {}".format(cpu_script_path))

    return model, torch.jit.load(cpu_script_path)


def dump_nl_binding_predictions(requests: List[NLBindingRequest], pipeline: NLBindingInferencePipeline,
                                saved_path: str):
    time_costs = []
    # wramup
    result = pipeline.infer(requests[0])

    in_list = []
    out_list = []

    saved_dir = os.path.dirname(saved_path)

    in_file = join(saved_dir, 'input.json')

    for request in tqdm(requests):
        result = pipeline.infer(request)
        time_costs.append(result.inference_ms)
        input_js = request.to_json()

        # export binding result json
        output_js = result.export_binding_json()

        in_list.append(input_js)
        out_list.append(output_js)

    save_json_file(in_file, in_list)
    save_json_file(saved_path, out_list)


def export_torch_to_model_bin(model_dir, model, dummy_input):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)
    parent_dir = os.path.dirname(model_dir)

    tokenizer_files = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', \
                       'sentencepiece.bpe.model', 'added_tokens.json', 'spm.phrase.txt']
    for tokenizer_file in tokenizer_files:
        shutil.copyfile(join(parent_dir, tokenizer_file), join(model_dir, tokenizer_file))

    # CPU model
    script_cpu_model = torch.jit.trace(model, dummy_input)
    cpu_model_path = os.path.join(model_dir, 'nl_binding.script.bin')
    torch.jit.save(script_cpu_model, cpu_model_path)


def infer_torchscript(args):
    checkpoint_dir = os.path.dirname(args.checkpoint)
    checkpoint_basename = os.path.basename(args.checkpoint)
    model_dir = join(checkpoint_dir, checkpoint_basename.replace('.pt', "_script"))

    # Load model
    model = GroundingInferenceModel.from_trained(args.checkpoint)

    # Load data inputs
    if args.sampling:
        data_inputs, data_requests, _ = load_data_inputs(args.data_path, tokenizer=model.tokenizer, sampling_size=200)
    else:
        data_inputs, data_requests, _ = load_data_inputs(args.data_path, tokenizer=model.tokenizer)

    dummy_input = data_inputs[0]
    print('sample data_inputs[0]  shape:')
    for input_name, input_value in zip(infer_input_names, dummy_input):
        print(input_name, " = ", input_value.size())
    print()

    if not os.path.exists(model_dir):
        export_torch_to_model_bin(model_dir, model, dummy_input)

    pipeline = NLBindingTorchScriptPipeline(model_dir, 0.2, use_gpu=torch.cuda.is_available() and args.gpu)
    # just try it
    result = pipeline.infer(data_requests[0])

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    predict_save_path = join(out_dir, 'predict.json')

    dump_nl_binding_predictions(data_requests, pipeline, predict_save_path)


if __name__ == '__main__':
    """
    python infer.py   -ckpt  checkpoint/UniG.step_15000.pt  -data_path data/wikisql_label/dev.preproc.json -out_dir output -gpu -sampling

    put below files in checkpoint folder:
    - added_tokens.json
    - config.json
    - model_config.json
    - sentencepiece.bpe.model
    - special_tokens_map.json
    - spm.phrase.txt
    - tokenizer_config.json

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint', type=str, required=True)
    parser.add_argument('-do_val', '--do_validation', action='store_true')
    parser.add_argument('-data_path', '--data_path', default='data/wikisql_label/dev.preproc.json')
    parser.add_argument('-out_dir', '--out_dir', default='./output')
    parser.add_argument('-sampling', '--sampling', action='store_true')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    args = parser.parse_args()

    infer_torchscript(args)
