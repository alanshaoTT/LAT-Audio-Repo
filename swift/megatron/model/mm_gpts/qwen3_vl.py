# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from contextlib import nullcontext
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import gpt_model
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from PIL import Image
from typing import List, Optional, Union

from swift.megatron.utils import split_cp_inputs
from swift.model import ModelType
from swift.utils import get_env_args, to_device
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge, MultimodalGPTBridge
from ..register import MegatronModelLoader, MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule

te_checkpoint = None

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint


class Qwen3Omni_Vit(HuggingFaceModule):
    module_mapping = {'thinker': 'thinker', 'talker': 'talker', 'code2wav': 'code2wav'}
    _vision_tower = ['thinker.audio_tower', 'thinker.visual']
    _aligner = [
        'thinker.audio_tower.proj1', 'thinker.audio_tower.proj2', 'thinker.visual.merger', 'thinker.visual.merger_list'
    ]
    _generator = ['talker', 'code2wav']

    def __init__(self, config):
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
        super().__init__(config, [Qwen3OmniMoeThinkerTextModel])

    def prepare_model(self, hf_model):
        del self.thinker.model
        del self.thinker.lm_head

    def _get_inputs_embeds(self, inputs_embeds, inputs, visual, processor, hf_config):
        input_ids = inputs['input_ids']
        packed_seq_params = inputs.get('packed_seq_params')
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            visual_res = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            if hasattr(visual_res, 'pooler_output'):
                image_embeds = visual_res.pooler_output
                deepstack_visual_embeds = visual_res.deepstack_features
            else:
                image_embeds, deepstack_visual_embeds = visual_res
            deepstack_visual_embeds = torch.stack(deepstack_visual_embeds, dim=0)
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            visual_pos_masks = None
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            visual_res = visual(pixel_values_mixed, grid_thw=grid_thw)
            if hasattr(visual_res, 'pooler_output'):
                mixed_embeds = visual_res.pooler_output
                deepstack_visual_embeds = visual_res.deepstack_features
            else:
                mixed_embeds, deepstack_visual_embeds = visual_res
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            image_mask = (input_ids == hf_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_mask = (input_ids == hf_config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if image_embeds is not None:
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            image_mask, video_mask = image_mask[..., 0], video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            if image_embeds is not None and video_embeds is not None:
                deepstack_image_embeds = [tensor[:image_tokens] for tensor in deepstack_visual_embeds]
                deepstack_video_embeds = [tensor[image_tokens:] for tensor in deepstack_visual_embeds]
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)

            deepstack_visual_embeds = torch.stack(deepstack_visual_embeds, dim=0)
            visual_pos_masks = visual_pos_masks.transpose(0, 1)
            # compat cp
            if self.config.context_parallel_size > 1:
                device = visual_pos_masks.device
                cp_mask = torch.full(visual_pos_masks.shape[:1], -1, dtype=torch.long, device=device)
                cp_mask[visual_pos_masks[:, 0]] = torch.arange(visual_pos_masks.sum(), device=device)
                cu_seqlens = getattr(packed_seq_params, 'cu_seqlens_q', None)
                cp_mask = split_cp_inputs(cp_mask, cu_seqlens, 0)
                visual_pos_masks = split_cp_inputs(visual_pos_masks, cu_seqlens, 0)
                deepstack_visual_embeds = deepstack_visual_embeds[:, cp_mask[(cp_mask != -1)]]
            # compat sp
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if self.config.sequence_parallel and tp_world_size > 1:
                visual_pos_masks = visual_pos_masks.view(tp_world_size, -1, *visual_pos_masks.shape[1:])
                mask_tokens = visual_pos_masks.sum(dim=(1, 2)).tolist()
                visual_start = 0 if tp_rank == 0 else sum(mask_tokens[:tp_rank])
                visual_end = visual_start + mask_tokens[tp_rank]
                visual_pos_masks = visual_pos_masks[tp_rank]
                deepstack_visual_embeds = deepstack_visual_embeds[:, visual_start:visual_end]
        return {
            'inputs_embeds': inputs_embeds,
            'visual_pos_masks': visual_pos_masks,
            'deepstack_visual_embeds': deepstack_visual_embeds
        }

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        # import pdb; pdb.set_trace()
        input_ids = kwargs['input_ids']
        visual = self.thinker.visual
        hf_config = self.hf_config.thinker_config
        res = self._get_inputs_embeds(inputs_embeds, kwargs, visual, self.processor, hf_config)
        inputs_embeds = res['inputs_embeds']
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')
        # print("actually get inputs_embeds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.thinker.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_res = self.thinker.get_audio_features(input_features, feature_attention_mask)
            if hasattr(audio_res, 'last_hidden_state'):
                audio_embeds = audio_res.last_hidden_state
            else:
                audio_embeds = audio_res
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_res = self.thinker.get_audio_features(input_features, feature_attention_mask)
            if hasattr(audio_res, 'last_hidden_state'):
                audio_embeds = audio_res.last_hidden_state
            else:
                audio_embeds = audio_res
            # // AIGC START: 根据环境变量，对每个样本的第一个 mp3 的 embedding 做 2/4/8 倍下采样
            factor = get_env_args('AUDIO_SEQ_COMP_FACTOR', int, 1)
            first_audio_token_len = kwargs.get('first_audio_token_len', None)               # [B]
            first_audio_token_len_orig = kwargs.get('first_audio_token_len_orig', None)     # [B]
            audio_token_lens_orig_list = kwargs.get('audio_token_lens_orig_list', None)     # list-of-lists，len = B
            
            # print(f"convvvvvvvvvvvvvvvvvvvvvvvvv {factor}")
            if (
                factor in (2, 4, 8)
                and isinstance(first_audio_token_len, torch.Tensor)
                and isinstance(first_audio_token_len_orig, torch.Tensor)
                and audio_token_lens_orig_list is not None
            ):
                # audio_embeds 按 get_audio_features 的实现，通常是 [T_total, D]
                if audio_embeds.dim() == 2:
                    T_total, D = audio_embeds.shape
                    # 修改为（增加一层判断，兼容嵌套 list）
                    total_token_lens_per_sample = []
                    for lens_list in audio_token_lens_orig_list:
                        # 如果 lens_list 依然是 list 嵌套 list，比如 [[100, 200]]
                        if len(lens_list) > 0 and isinstance(lens_list[0], list):
                            # 展平一层
                            flat_list = [item for sublist in lens_list for item in sublist]
                            total_token_lens_per_sample.append(int(sum(flat_list)))
                        else:
                            total_token_lens_per_sample.append(int(sum(lens_list)))
                    # # 每个样本所有 mp3 的原始 token 总数：total_i = sum(audio_token_lens_orig_list[i])
                    # total_token_lens_per_sample = [
                    #     int(sum(lens_list)) for lens_list in audio_token_lens_orig_list
                    # ]
                    B = len(total_token_lens_per_sample)
                    # 与长度信息不一致就直接跳过压缩，避免训练崩掉
                    if first_audio_token_len.numel() != B or first_audio_token_len_orig.numel() != B:
                        pass
                    elif sum(total_token_lens_per_sample) != T_total:
                        # 长度对不上，谨慎起见不做压缩
                        pass
                    else:
                        # 计算每个样本在大序列中的起止下标 offsets: [0, total_0, total_0+total_1, ...]
                        offsets = [0]
                        for total in total_token_lens_per_sample:
                            offsets.append(offsets[-1] + total)
                        new_segments = []
                        for i in range(B):
                            start = offsets[i]
                            end = offsets[i + 1]
                            if start >= T_total:
                                break
                            end = min(end, T_total)
                            seg = audio_embeds[start:end, :]  # 该样本在大序列中的整段 [Ti, D]
                            # 取出该样本第一个 mp3 的压缩前/压缩后 token 数（仍在同一 device，上面没 to(cpu)）
                            L1_i = int(first_audio_token_len[i].item())
                            L1_orig_i = int(first_audio_token_len_orig[i].item())
                            # 只在合法区间内做插值压缩
                            if 1 <= L1_i < L1_orig_i <= seg.size(0):
                                first_seg = seg[:L1_orig_i, :]      # 当前样本第一个 mp3 对应的 token 段 [L1_orig_i, D]
                                rest_seg = seg[L1_orig_i:, :]       # 该样本剩余 token
                                # 线性插值把 L1_orig_i 压缩到 L1_i
                                x = first_seg.t().unsqueeze(0)      # [1, D, L1_orig_i]
                                # print("发生了压缩！！！！！！！！！！！")
                                x = torch.nn.functional.interpolate(
                                    x,
                                    size=L1_i,
                                    mode='linear',
                                    align_corners=False,
                                )
                                first_seg_ds = x.squeeze(0).t()     # [L1_i, D]
                                seg_new = torch.cat([first_seg_ds, rest_seg], dim=0)  # [L1_i + Ti - L1_orig_i, D]
                            else:
                                seg_new = seg
                            new_segments.append(seg_new)
                        if len(new_segments) > 0:
                            audio_embeds = torch.cat(new_segments, dim=0)  # 仍然是 [T_total_new, D]
                else:
                    # 极少数不是 2D 的情况，暂时跳过压缩，保持原行为
                    pass
            # // AIGC END
            audio_mask = (input_ids == hf_config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        res['inputs_embeds'] = inputs_embeds
        return res


class Qwen3VLTransformerBlock(gpt_model.TransformerBlock):
    # Code borrowed from NVIDIA/Megatron-LM

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_bias: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_fp8_context: bool,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):

            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb, visual_pos_masks,
                               deepstack_visual_embeds):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number
                                        - 1) if use_inner_fp8_context else nullcontext())
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                    # add visual features to the hidden states of first several layers
                    layer_number = layer.layer_number - 1
                    if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_number],
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers))

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (layer_idx >= recompute_skip_num_layers
                        and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(hidden_states, attention_mask, context,
                                                                              context_mask, rotary_pos_emb)
        else:
            raise ValueError('Invalid activation recompute method.')

        return hidden_states

    def forward(
        self,
        hidden_states: Union[torch.Tensor, WrappedTensor],
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ):
        """
        Perform the forward pass through the transformer block.
        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.
        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        if deepstack_visual_embeds is not None:
            assert len(deepstack_visual_embeds) <= len(
                self.layers), (f'len(deepstack_visual_embeds): {len(deepstack_visual_embeds)}, '
                               f'len(self.layers): {len(self.layers)}.')
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_fp8_context=use_inner_fp8_context,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number
                                        - 1) if use_inner_fp8_context else nullcontext())
                    with self.offload_context, inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    # add visual features to the hidden states of first several layers
                    layer_number = layer.layer_number - 1
                    if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_number],
                        )

                    if (torch.is_grad_enabled() and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        if visual_pos_masks is None:
            return hidden_states + visual_embeds.mean() * 0
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3OmniBridge(GPTBridge):
    hf_layers_prefix = 'thinker.model.layers'
    hf_embed_key = 'thinker.model.embed_tokens.weight'
    hf_final_layernorm_key = 'thinker.model.norm.weight'
    hf_lm_head_key = 'thinker.lm_head.weight'
    hf_score_key = 'thinker.score.weight'


class Qwen3VL_Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger', 'visual.deepstack_merger_list']

    def __init__(self, config):
        from transformers.models.qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeTextModel
        super().__init__(config, [Qwen3VLTextModel, Qwen3VLMoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Qwen3Omni_Vit._get_inputs_embeds(self, inputs_embeds, kwargs, self.visual, self.processor,
                                                self.hf_config)


class Qwen3VLLoader(MegatronModelLoader):

    def _patch_transformer_block(self):
        if hasattr(gpt_model, 'OriginTransformerBlock'):
            return
        gpt_model.OriginTransformerBlock = gpt_model.TransformerBlock
        gpt_model.TransformerBlock = Qwen3VLTransformerBlock

    def __init__(self, args, hf_config):
        super().__init__(args, hf_config)
        self._patch_transformer_block()


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_vl,
        [
            ModelType.qwen3_vl,
            ModelType.qwen3_vl_moe,
            ModelType.qwen3_vl_emb,
            ModelType.qwen3_vl_reranker,
        ],
        bridge_cls=MultimodalGPTBridge,
        visual_cls=Qwen3VL_Vit,
        loader=Qwen3VLLoader,
    ))

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_omni,
        [
            ModelType.qwen3_omni_moe,
        ],
        bridge_cls=Qwen3OmniBridge,
        visual_cls=Qwen3Omni_Vit,
        loader=Qwen3VLLoader,
    ))
