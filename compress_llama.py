# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from typing import Any, Dict, List, Union
import torch
import bmtrain as bmt
import os
from arguments import get_args

from model_center.model import Llama, LlamaConfig
from model_center.tokenizer import LlamaTokenizer
from training_task import MixedDataset
from model_center.layer import Linear, TransformerBlock, Attention, FeedForward
import math
import bmcook
from bmcook import CookTrainer, ConfigParser
import logging

import fnmatch
from bmtrain import nccl


from bmcook.pruning import *
from bmcook.utils import config as bmcook_config

class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))


def compress_setup(args, model, optimizer):
    # teacher = get_model(args, True)
    teacher = None
    cook_config = ConfigParser(args.cook_config)
    CookTrainer.set_compression(cook_config, model, optimizer, teacher=teacher, quant_layer_cls=Linear)

def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_config)
    return tokenizer


def get_model(args, teacher: bool =False):
    config_path = args.model_config if not teacher else args.teacher_config
    config = LlamaConfig.from_pretrained(config_path)
    if args.cook_save_mode in ["moefication", "mix", "relu"]:
        config.ffn_activate_fn = "relu"
    model = Llama(config)
    model.config = config
    if args.load is not None:
        if teacher:
            bmt.load(model, args.load_teacher)
        else:
            bmt.load(model, args.load, strict=False)
    else:
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model: torch.nn.Module):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    if args.load is not None:
        if os.path.exists(os.path.join(args.save, args.save_name + (".rank-%d.opt" % 0))):
            # optimizer state exists
            states = torch.load(
                os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank()))
            )
            optimizer.load_state_dict(states)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_scheduler = Cosine(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    optim_manager = bmt.optim.OptimManager(
        loss_scale=args.loss_scale,
        loss_scale_factor=2,
        loss_scale_steps=32,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return tokenizer, model, optimizer, lr_scheduler, optim_manager


def initialize():
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2333)
    args = get_args(pretrain=True, compress=True)
    bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


class LossSpikeDetector:
    def __init__(self, log_path: str) -> None:
        self._last_loss: Dict[str, float] = {}
        self._last_data: List[Any] = [None]
        self._log_path = log_path

    def update_data(self, data: Any):
        self._last_data.append(data)
        if len(self._last_data) > 2:
            self._last_data = self._last_data[-2:]

    def update_loss(self, iteration: int, loss_map: Dict[str, float]):
        loss_spike_result = []
        for task, loss in loss_map.items():
            if task in self._last_loss:
                if loss > self._last_loss[task] * 3:
                    # loss spike!
                    loss_spike_result.append(
                        {
                            "prev": self._last_loss[task],
                            "curr": loss,
                            "task": task,
                        }
                    )
            self._last_loss[task] = float(loss)
        if len(loss_spike_result) > 0:
            self._write_log(iteration, self._last_data[-1], loss_spike_result)

    def _write_log(self, iteration: int, data: Any, result: List[Dict[str, Any]]):
        with open(self._log_path, "a", encoding="utf-8") as fp:
            fp.write("=" * 20)
            fp.write("\nloss spike at {}\n".format(iteration))
            fp.write("{}\n".format(json.dumps(result, indent=4, ensure_ascii=False)))
            fp.write("data: \n")
            for d in data:
                fp.write("{}\n".format(json.dumps(d, indent=4, ensure_ascii=False)))
            fp.write("\n\n")


def compress(
    args,
    tokenizer: LlamaTokenizer,
    model: Llama,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    optim_manager: bmt.optim.OptimManager,
    is_moefy: bool = False,
):

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger(__name__)

    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step

    lsd = LossSpikeDetector("debug/spile.%d.log" % bmt.rank())

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()
    dataloader = MixedDataset(
        args.dataset, args.batch_size, args.max_length, tokenizer, max_depth=8
    )

    if os.path.exists(os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))):
        # load dataset states if exists
        dataset_states = torch.load(
            os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))
        )
        missing = dataloader.load_state_dict(dataset_states)
        if len(missing) > 0:
            bmt.print_rank("Missing keys when loading dataset states: ", missing)
    dataloader.start()
    prune_iters = 500
    block_index = 1
    
    if is_moefy:
        os.makedirs(args.cook_save + '/hiddens', exist_ok=True)
        model.eval()

        hiddens_dir = args.cook_save + '/hiddens'
        os.makedirs(hiddens_dir, exist_ok=True)

        for iteration, data in enumerate(dataloader):

            if iteration == 100:
                break

            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            dec_mask = torch.arange(args.max_length)[None, :].repeat(args.batch_size, 1) < input_length[:, None]
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            
            with torch.no_grad():
                outputs = CookTrainer.forward(model, loss_func, targets, input_ids, input_length)
            
            torch.save(outputs[-1], args.cook_save + '/hiddens/{}_{}'.format(iteration, bmt.rank()))
               
            bmt.print_rank("Iteration:", iteration)
        exit()

    try:
        for iteration, data in enumerate(dataloader):

            iteration = iteration + start_step + 1
            assert data["inputs"].shape[0] == args.batch_size
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            lsd.update_data(data["raw_data"])

            # ===========
            optim_manager.zero_grad()
            # torch.cuda.empty_cache()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            outputs = CookTrainer.forward(model, loss_func, targets,
                input_ids,
                input_length,
            )
            logits = outputs.original_output[0]
            loss = outputs.loss
            d_loss = bmt.sum_loss(outputs.d_loss).item() if outputs.d_loss != 0 else 0
            # logits = model(input_ids, input_length).logits
            # loss = loss_func(logits.view(-1, logits.shape[-1]), targets.view(-1))
            # d_loss = 0.
            # for i in range(100): print(logits[0][i][targets[0][i]], logits[0][i].max())
            # if bmt.rank()==0:from IPython import embed; embed()
            # bmt.synchronize()
            global_loss = bmt.sum_loss(loss).item()
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # ===========
            current_stream = torch.cuda.current_stream()
            # some reduce ops of distributed parameter were launched on load stream
            current_stream.wait_stream(bmt.config['load_stream'])
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

            # ==========
            iteration_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iteration_time)

            # with torch.no_grad():
            #     task_num = len(task_names)
            #     targets_tmp = targets.expand(task_num, -1, -1)
            #     task = torch.arange(task_num, dtype=torch.int32, device="cuda")[:, None, None]
            #     targets_tmp = torch.where(
            #         task_ids == task,
            #         targets_tmp,
            #         torch.scalar_tensor(-100, dtype=torch.int32, device="cuda"),
            #     )

            #     task_loss_map: Dict[str, float] = {}
            #     for i in range(task_num):
            #         task_loss = loss_func(
            #             logits.view(-1, logits.size(-1)), targets_tmp[i, :].view(-1)
            #         )
            #         # global_task_loss = float(bmt.sum_loss(task_loss).item())
            #         task_loss_map[task_names[i]] = task_loss.item()
            #     gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)

            #     global_task_loss_map: Dict[str, Union[List[float], float]] = {}
            #     for local_task_loss_map in gatherd_task_loss_map:
            #         for task_name, task_loss in local_task_loss_map.items():
            #             if task_name not in global_task_loss_map:
            #                 global_task_loss_map[task_name] = []
            #             global_task_loss_map[task_name].append(task_loss)

            #     task_loss_map = {}
            #     for task_name in sorted(list(global_task_loss_map.keys())):
            #         avg_loss = sum(global_task_loss_map[task_name]) / len(
            #             global_task_loss_map[task_name]
            #         )
            #         task_loss_map[task_name] = avg_loss

            local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += (
                global_world_size * local_total_rate * args.max_length * args.batch_size
            )
            avg_time = average_time.value
            # lsd.update_loss(iteration, task_loss_map)

            train_info = {
                "time": tim_usage["init"],
                "iteration": iteration,
                "loss": global_loss,
                "lr": lr_scheduler.current_lr,
                "lr_scale": int(optim_manager.loss_scale),
                "time_usage": tim_usage,
                "mem_usage": mem_usage,
                "avg_time": avg_time,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / avg_time,
                "grad_norm": grad_norm.item(),
                "mask_max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
                "num_gpus": global_world_size,
                # "task_loss": task_loss_map,
            }

            bmt.print_rank(
                (
                    "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f} | d_loss: {:.4f}"
                ).format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    avg_time,
                    input_length.float().mean() / args.max_length,
                    (targets >= 0).sum(-1).float().mean() / args.max_length,
                    grad_norm,
                    d_loss
                )
            )
            # bmt.print_rank(
            #     "| "
            #     + " | ".join(
            #         [
            #             "{} loss: {:.4f}".format(task_name, loss)
            #             for task_name, loss in task_loss_map.items()
            #         ]
            #     )
            #     + "\n"
            # )

            if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                # bmt.print_rank(bmt.inspect.format_summary(model_inspect))
                train_info["model_inspect"] = model_inspect

            # write log here
            # if args.log_dir is not None and bmt.rank() == 0:
            #     log_mgr.write(**train_info)
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                # for task_name, loss in task_loss_map.items():
                #     writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)
                writer.add_scalar("Compress/d_loss", d_loss, iteration)
                writer.add_scalar("Compress/Grad_norm", grad_norm, iteration)
                writer.add_scalar("Compress/Loss_scale", int(optim_manager.loss_scale), iteration)

                if iteration % args.inspect_iters == 0:
                    text_summray = bmt.inspect.format_summary(model_inspect)
                    logger.debug(text_summray)

            if iteration % args.cook_save_iters == 0: #or iteration % args.cook_save_mask_iters == 0:
                # bmcook.save_masks(args.cook_mask_save)
                # bmcook.save(model, os.path.join(args.cook_save, args.cook_save_name + "_cook" + ("-%d" % iteration)), args.cook_save_mode)
                bmt.save(model, os.path.join(args.cook_save, args.cook_save_name + ("-%d" % iteration)))

            # if args.save is not None and iteration % args.save_iters == 0:
            #     bmt.save(model, os.path.join(args.save, args.save_name + ("-%d.pt" % iteration)))
            #     torch.save(
            #         optimizer.state_dict(),
            #         os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank())),
            #     )
            #     all_states = dataloader.state_dict()
            #     if bmt.rank() == 0:
            #         # rank 0 writes the dataloader state
            #         torch.save(
            #             all_states,
            #             os.path.join(args.save, args.save_name + ("-%d.data.pt" % iteration)),
            #         )
            #     del all_states
    finally:
        dataloader.close()



def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    compress_setup(args, model, optimizer)

    compress(args, tokenizer, model, optimizer, lr_scheduler, optim_manager, args.cook_save_mode == 'moe')


if __name__ == "__main__":
    main()
