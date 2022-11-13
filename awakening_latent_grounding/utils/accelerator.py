import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

is_amp_available = True
try:
    import apex
    from apex import amp
except ImportError:
    is_amp_available = False

def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    # There is no direct check whether an object if of type namedtuple sadly, this is a workaround.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # Can instantiate a namedtuple from a generator directly, contrary to a tuple/list.
        return type(obj)(*list(generator))
    return type(obj)(generator)

def convert_to_fp32(tensor):
    """
    Recursively converts the lements nested list/tuple/dictionary of tensors in FP16 precision to FP32.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to convert from FP16 to FP32.

    Returns:
        The same data structure as :obj:`tensor` with all tensors that were in FP16 precision converted to FP32.
    """
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (convert_to_fp32(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: convert_to_fp32(v) for k, v in tensor.items()})
    elif not hasattr(tensor, "dtype") or tensor.dtype != torch.float16:
        return tensor
    return tensor.float()


def convert_outputs_to_fp32(model_forward):
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP16
    precision will be convert back to FP32.

    Args:
        model_forward (:obj:`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as :obj:`model_forward` but with converted outputs.
    """

    def convert_outputs(*args, **kwargs):
        outputs = model_forward(*args, **kwargs)
        return convert_to_fp32(outputs)

    return convert_outputs

"""
Accelerator for distributed training
"""
class DistributedAccelerator:
    use_fp16: bool
    fp16_opt_level: str

    # distributed training
    word_size: int
    rank_index: int
    local_rank_index: int

    device: torch.device

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, use_fp16: bool, fp16_opt_level=None, use_cpu: bool= False) -> None:
        self.use_fp16 = use_fp16
        self.fp16_opt_level = fp16_opt_level

        if int(os.environ.get("LOCAL_RANK", -1)) != -1 and not use_cpu:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            self.word_size = torch.distributed.get_world_size()
            self.rank_index = torch.distributed.get_rank()
            self.local_rank_index = int(os.environ.get("LOCAL_RANK", -1))

            self.device = torch.device("cuda", self.local_rank_index)
            torch.cuda.set_device(self.device)
        else:
            self.word_size = 1
            self.rank_index = -1
            self.local_rank_index = -1

            if use_cpu:
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
        pass

    def __repr__(self):
        repr = (
            f"Word Size: {self.word_size}\n"
            f"Rank index: {self.rank_index}\n"
            f"Local Rank index: {self.local_rank_index}\n"
            f"Device: {self.device}\n"
            f"Use FP16 precision: {self.use_fp16}\n"
        )
        return repr

    @property
    def is_local_main_process(self):
        return self.local_rank_index in [-1, 0] # -1 means single gpu

    @property
    def is_distributed(self):
        return self.local_rank_index != -1

    def wrapper_model_and_optimizer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        model = model.to(self.device)
        if self.use_fp16:
            if not is_amp_available:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level if self.fp16_opt_level else "O1")

        if self.is_distributed:
            if self.use_fp16:
                model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank_index],
                    output_device=self.local_rank_index,
                    find_unused_parameters=True
                )

        self.model = model
        self.optimizer = optimizer
        return model, optimizer

    def wrapper_data_loader(self, data_loader: DataLoader) -> DataLoader:
        if not self.is_distributed:
            return data_loader

        dataset: Dataset = data_loader.dataset
        return DataLoader(dataset,
            sampler=DistributedSampler(dataset, rank=self.local_rank_index),
            batch_size=data_loader.batch_size,
            collate_fn=data_loader.collate_fn)

    def backward(self, loss: torch.Tensor):
        if self.use_fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def clip_grad_norm(self, max_grad_norm: float):
        if self.use_fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

    def wait_for_everyone(self):
        if self.is_distributed:
            torch.distributed.barrier()

    """
    def gather(self, data):
        if isinstance(data, torch.Tensor):
            return self._gather_one(data)
        elif isinstance(data, (tuple, list)):
            return [self.gather(x) for x in data]
        elif isinstance(data, dict):
            return { k : self.gather(v) for k, v in data.items() }
        else:
            return None

    def _gather_one(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim == 0:
            input_tensor = input_tensor.clone()[None]
            raise ValueError()

        output_tensors = [input_tensor.clone() for _ in range(torch.distributed.get_world_size())]
        print("Gather inputs", self.local_rank_index, input_tensor.size())

        torch.distributed.all_gather(output_tensors, input_tensor)
        for output_tensor in output_tensors:
            print(output_tensor.size())
        output_tensors = torch.cat(output_tensors, dim=0)
        print("Gather outputs", self.local_rank_index, output_tensors.size())
        return output_tensors
    """
