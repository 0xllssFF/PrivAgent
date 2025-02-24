# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import bitsandbytes
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from aiolimiter import AsyncLimiter
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import json
from transformers import (
    Adafactor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    LogitsProcessor,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from project_env import PROMPT_PATH
from rewards.text_rewards import TextRewards



def get_logger(name):
    if 'log' in name:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    elif 'prompt' in name:
        class CSVMessageFormatter(logging.Formatter):
            def format(self, record):
                message = record.getMessage()
                message = message.replace('"', '""')
                return f'"{message}"'
            
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(name)
        file_handler.setLevel(logging.INFO)
        csv_message_formatter = CSVMessageFormatter()
        file_handler.setFormatter(csv_message_formatter)

        logger.addHandler(file_handler)
    
    return logger


class TwoStageLogitsProcessor(LogitsProcessor):
    def __init__(self, stage_length: int, first_temp: float, second_temp: float):
        self.stage_length = stage_length
        self.first_temp = first_temp
        self.second_temp = second_temp

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        curr_length = input_ids.shape[1]
        if curr_length <= self.stage_length:
            return scores / self.first_temp

        return scores / self.second_temp


def run_victim_model(client, messages, model, tok, limiter):
    if client:
        if "gpt" in script_args.victim_model:
            resp = asyncio.run(
                openai_batch_async_chat_completion(
                    messages, client=client, model=model, limiter=limiter
                )
            )
            outputs = [
                output.choices[0].message.content if output else "" for output in resp
            ]
        elif "claude" in script_args.victim_model:
            resp = asyncio.run(
                claude_batch_async_chat_completion(
                    messages, client=client, model=model, limiter=limiter
                )
            )
            outputs = [output.content[0].text if output else "" for output in resp]
        else:
            resp = asyncio.run(
                openai_batch_async_chat_completion(
                    messages, client=client, model=model, limiter=limiter
                )
            )
            outputs = [
                output.choices[0].message.content if output else "" for output in resp
            ]
        resps = outputs
        # print if there is any empty response
        if not all(resps):
            print("empty response detected")
    else:
        terminator = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        inputs_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output = model.generate(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            eos_token_id=terminator,
            pad_token_id=tok.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[:, inputs_ids.shape[1] :].cpu()
        resps = tok.batch_decode(output, skip_special_tokens=True)
    return resps


async def chat_function(
    chat, model, messages, temperature=0.6, top_p=0.9, max_tokens=128
):
    for i in range(5):
        # sleep for a while to avoid rate limit
        try:
            if "claude" in model:
                # extract system message
                system_message = [
                    message["content"]
                    for message in messages
                    if message["role"] == "system"
                ][0]
                user_messages = [
                    message for message in messages if message["role"] != "system"
                ]
                ret = await chat(
                    model=model,
                    system=system_message,
                    messages=user_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            else:
                ret = await chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            return ret
        except Exception as e:
            print(f"failed with error {e}, retrying")
            await asyncio.sleep(10)
            continue
    return None


async def openai_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(
            limiter, messages, model, client.chat.completions.create
        )
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks)


async def claude_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(limiter, messages, model, client.messages.create)
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks)


async def rate_limited_api_call_precise(limiter, messages, model, llm_func):
    async with limiter:
        return await chat_function(
            chat=llm_func,
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.6,
            top_p=0.9,
        )


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "the tokenizer name"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    target_dataset: Optional[str] = field(
        default="train_data_pleak.csv",
    )
    wandb_exp_name: Optional[str] = field(
        default="default", metadata={"help": "the wandb experiment name"}
    )
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "the wandb entity"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=35, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=2, metadata={"help": "the PPO minibatch size"}
    )
    score_sample_num: Optional[int] = field(
        default=5, metadata={"help": "the number of samples for scoring"}
    )
    # for victim model
    server_url: Optional[str] = field(
        default="", metadata={"help": "the server url for victim model"}
    )
    api_key: Optional[str] = field(
        default="", metadata={"help": "the api key for victim model"}
    )
    victim_model: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "the victim model"},
    )
    requests_per_minute: Optional[int] = field(
        default=5, metadata={"help": "requests per minute"}
    )

    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    use_bonus_reawrd: Optional[bool] = field(
        default=True, metadata={"help": "whether to use bonus reward"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    resume: Optional[bool] = field(
        default=False, metadata={"help": "whether to resume training"}
    )
    resume_checkpoint: Optional[str] = field(
        default="", metadata={"help": "the checkpoint to resume training"}
    )
    save_freq: Optional[int] = field(
        default=1, metadata={"help": "n steps to save the model"}
    )
    output_dir: Optional[str] = field(
        default="batch_32_runs/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=40000, metadata={"help": "number of epochs"})
    epochs: Optional[int] = field(default=100, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.01,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )

    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    use_score_norm: Optional[bool] = field(
        default=True, metadata={"help": "whether to normalize the score"}
    )
    use_score_scaling: Optional[bool] = field(
        default=True, metadata={"help": "whether to scale the score"}
    )
    # threshold for good prompts
    reward_threshold: Optional[float] = field(
        default=0.9,
        metadata={
            "help": "reward threshold for good prompts. Only prompts with avg reward higher than this threshold will be saved"
        },
    )
    similarity_threshold: Optional[float] = field(
        default=0.75,
        metadata={
            "help": "threshold for similar prompts. Only prompts with similarity compared to previous saved prompts lower than this threshold will be saved"
        },
    )

    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 8bit"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 4bit"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "use gradient checkpointing"}
    )
    # for 4bit quantization
    use_nested_quant: Optional[bool] = field(
        default=True, metadata={"help": "whether to use nested quant"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type fp4 or nf4"}
    )
    exp_exp_threshold: float = field(
        default=0.3,
        metadata={
            "help": "threshold for explore-exploit. change the temperature stages."
        },
    )
    enable_sampling: Optional[bool] = field(
        default=False, metadata={"help": "enable sampling"}
    )


def build_dataset(tokenizer):
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for query in examples["questions"]:
            tokenized_question = tokenizer(query, padding=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


if __name__ == "__main__":
    load_dotenv()
    global_good_prompts_dict = {"text": [], "reward": []}
    accelerator = Accelerator()
    current_device = accelerator.local_process_index
    # torch.multiprocessing.set_start_method('spawn')
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    os.makedirs(script_args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(script_args.output_dir, "log.txt"))

    ### add prompt tracer
    
    prompt_logger = get_logger(os.path.join(script_args.output_dir, "prompt.txt"))


    config = PPOConfig(
        steps=script_args.steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        tracker_kwargs={
            "wandb": {
                "entity": script_args.wandb_entity,
                "name": script_args.wandb_exp_name,
            }
        },
        tracker_project_name="rl_finetune_llm",
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        use_score_norm=script_args.use_score_norm,
        use_score_scaling=script_args.use_score_scaling,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed + current_device,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        gradient_checkpointing=script_args.gradient_checkpointing,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name, padding_side="left"
    )
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create the dataset
    training_num = 1000
    train_dataset = Dataset.from_dict(
        {
            "questions": ["Please generate a prompt for me: "] * training_num,
        }
    )
    original_columns = train_dataset.column_names

    dataset = build_dataset(tokenizer)
    training_targets = pd.read_csv(
        os.path.join(PROMPT_PATH, script_args.target_dataset)
    )["text"].tolist()
    # with open(os.path.join(PROMPT_PATH, "chatgpt.txt"), "r") as f:
    #     chatgpt_prompt = f.read()
    # with open(os.path.join(PROMPT_PATH, "claude.txt"), "r") as f:
    #     claude_prompt = f.read()
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if script_args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_compute_dtype=torch.float16
        )
    elif script_args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=script_args.use_nested_quant,
        )
    else:
        raise ValueError("Please specify a quantization config")
    if script_args.resume:
        current_epoch = int(script_args.resume_checkpoint.split("_")[-1]) + 1
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.resume_checkpoint,
            quantization_config=quant_config,
            device_map={"": current_device},
            low_cpu_mem_usage=True if quant_config is not None else False,
            trust_remote_code=True,
        )
    else:
        current_epoch = 0
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            device_map={"": current_device},
            peft_config=lora_config,
            low_cpu_mem_usage=True if quant_config is not None else False,
            trust_remote_code=True,
        )

    optimizer = bitsandbytes.optim.PagedAdamW32bit(
        model.parameters(), lr=config.learning_rate
    )
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )
    limiter, client, victim_model, victim_tokenizer = None, None, None, None
    if "gpt" in script_args.victim_model:
        import openai

        client = openai.AsyncOpenAI()
        victim_model = script_args.victim_model
        limiter = AsyncLimiter(script_args.requests_per_minute, 60)
    elif "claude" in script_args.victim_model:
        import anthropic

        client = anthropic.AsyncAnthropic()
        victim_model = script_args.victim_model
        limiter = AsyncLimiter(script_args.requests_per_minute, 60)
    elif script_args.server_url:
        import openai

        client = openai.AsyncOpenAI(
            base_url=script_args.server_url,
            api_key=script_args.api_key,
        )
        victim_model = script_args.victim_model
        limiter = AsyncLimiter(script_args.requests_per_minute, 60)
    else:
        victim_model = AutoModelForCausalLM.from_pretrained(
            script_args.victim_model,
            device_map={"": current_device},
            quantization_config=quant_config,
        ).eval()
        victim_tokenizer = AutoTokenizer.from_pretrained(
            script_args.victim_model, padding_side="left"
        )
        victim_tokenizer.pad_token = victim_tokenizer.eos_token

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        # "eos_token_id": 100_000,
    }
    output_min_length = 15
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    def score(prompts, model, tok, client, limiter, sample_num=6):
        rewards = []
        messages = []
        # for system prompt
        eval_targets = random.sample(training_targets, sample_num)
        # add another two prompts
        # eval_targets.extend([claude_prompt])
        for prompt in prompts:
            for text in eval_targets:
                messages.append(
                    [
                        {"role": "system", "content": text},
                        {"role": "user", "content": prompt},
                    ]
                )
        resps = run_victim_model(client, messages, model, tok, limiter)
        good_prompts = []
        good_rewards = []
        for i in range(len(prompts)):
            current_resps = resps[i * len(eval_targets) : (i + 1) * len(eval_targets)]
            # remove empty response and filter targets accordingly
            current_eval_targets = [
                eval_targets[j] for j in range(len(eval_targets)) if current_resps[j]
            ]
            current_resps = [resp for resp in current_resps if resp]
            assert len(current_resps) == len(
                current_eval_targets
            ), "response and target should have the same length"
            if not current_resps:
                # for this one we need to rerun the generation
                for _ in range(5):
                    empty_messages = messages[
                        i * len(eval_targets) : (i + 1) * len(eval_targets)
                    ]
                    empty_resps = run_victim_model(
                        client, empty_messages, model, tok, limiter
                    )
                    current_eval_targets = [
                        eval_targets[j]
                        for j in range(len(eval_targets))
                        if empty_resps[j]
                    ]
                    current_resps = [resp for resp in empty_resps if resp]
                    if current_resps:
                        break
                assert current_resps, "response is still empty after rerun"
            # reward
            reward = np.mean(
                TextRewards.distance_lcs(current_resps, current_eval_targets)
            )
            # bonus reward
            if script_args.use_bonus_reawrd and global_good_prompts_dict["text"]:
                nums = len(global_good_prompts_dict["text"])
                bonus_reward = np.max(
                    TextRewards.distance_lcs(
                        [prompts[i]] * nums, global_good_prompts_dict["text"]
                    )
                )
                final_reward = 0.8 * reward + 0.2 * (1 - bonus_reward)
            else:
                final_reward = reward
            if reward > script_args.reward_threshold:
                good_prompts.append(prompts[i])
                good_rewards.append(reward)
            rewards.append(final_reward)
            if accelerator.is_main_process:
                logger.info(f"{prompts[i]}")
                logger.info(f"{reward}")
                prompt_logger.info(prompts[i].replace('\n', '\\n'))
        if accelerator.is_main_process:
            logger.info("--------------------------------------")
        if not good_prompts:
            good_prompts = [0]
            good_rewards = [0]
        # for diversity
        all_good_prompts = accelerator.gather_for_metrics(good_prompts)
        all_good_rewards = accelerator.gather_for_metrics(good_rewards)

        # filter out the empty prompt
        all_good_prompts = [prompt for prompt in all_good_prompts if prompt]
        all_good_rewards = [reward for reward in all_good_rewards if reward]
        all_good_data = list(zip(all_good_prompts, all_good_rewards))
        if all_good_data:
            all_good_data = list(set(all_good_data))

            if not global_good_prompts_dict["text"]:
                global_good_prompts_dict["text"].append(all_good_data[0][0])
                global_good_prompts_dict["reward"].append(all_good_data[0][1])
                all_good_data = all_good_data[1:]
            for prompt, reward in all_good_data:
                # remove similar prompts
                similarity = TextRewards.distance_lcs(
                    [prompt] * len(global_good_prompts_dict["text"]),
                    global_good_prompts_dict["text"],
                )
                max_similarity = np.max(similarity)
                if max_similarity < script_args.similarity_threshold:
                    global_good_prompts_dict["text"].append(prompt)
                    global_good_prompts_dict["reward"].append(reward)
                # global_good_prompts_dict["text"].append(prompt)
                # global_good_prompts_dict["reward"].append(reward)

        return [torch.FloatTensor([score]) for score in rewards]

    exp_exp_stage = 0
    for epoch in tqdm(
        range(current_epoch, script_args.epochs),
        "epoch: ",
        initial=current_epoch,
        total=script_args.epochs,
    ):
        for batch in tqdm(ppo_trainer.dataloader):
            if epoch >= config.total_ppo_epochs:
                break

            question_tensors = batch["input_ids"]
            if script_args.enable_sampling:
                logger.info("-------sampling enabled-------")
                response_tensors = ppo_trainer.generate(
                    question_tensors,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    temperature=1.0,
                    logits_processor=[
                        # TODO: explore-exploit stages
                        TwoStageLogitsProcessor(
                            len(question_tensors[0]),
                            32 if exp_exp_stage == 0 else 0.6,
                            1.0 if exp_exp_stage == 0 else 0.6,
                        )
                    ],
                    **generation_kwargs,
                )
            else:
                response_tensors = ppo_trainer.generate(
                    question_tensors,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    **generation_kwargs,
                )
            response_tensors = list(response_tensors)
            batch["response"] = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )

            # Compute reward score
            accelerator.print("-------start calculating reward-------")
            rewards = score(
                batch["response"],
                model=victim_model,
                tok=victim_tokenizer,
                client=client,
                sample_num=script_args.score_sample_num,
                limiter=limiter,
            )
            accelerator.print("-------rewards calculated-------")
            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            if (
                script_args.enable_sampling
                and exp_exp_stage == 0
                and np.max(rewards) > script_args.exp_exp_threshold
            ):
                logger.info("-------explore-exploit stage 1-------")
                exp_exp_stage = 1
        if (
            script_args.save_freq and epoch and epoch % script_args.save_freq == 0
        ) or epoch == script_args.epochs - 1:
            ppo_trainer.save_pretrained(
                os.path.join(script_args.output_dir, f"epoch_{epoch}")
            )
        # save good prompts to csv
        if accelerator.is_main_process:
            pd.DataFrame(global_good_prompts_dict).to_csv(
                os.path.join(script_args.output_dir, "good_prompts.csv"), index=False
            )
            accelerator.print("-------good prompts saved-------")
