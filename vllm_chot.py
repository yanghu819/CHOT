
import os
class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input = \
            ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, model_input.seq_lens,
                model_input.query_lens, self.device, self.pin_memory,
                generators, self.sampling_metadata_cache)
        else:
            sampling_metadata = None
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    @dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    kv_caches=kv_caches,
                    attn_metadata=model_input.attn_metadata,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                 device=self.device),
                    **seqlen_agnostic_kwargs)

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states
        

        if model_input.is_prompt:
            print("DEBUG - Prompt phase detected")
            chot_optimized = os.environ.get("CHOT_OPTIMIZED", "0") == "1"
            print(f"DEBUG - CHOT_OPTIMIZED: {chot_optimized}")
            
            # 只在未优化过时执行优化
            if not chot_optimized:
                print("DEBUG - Starting CHOT optimization")
                chot_steps = int(os.environ.get("CHOT_STEPS", "3"))
                chot_lr = float(os.environ.get("CHOT_LR", "1e-4"))
                print(f"DEBUG - CHOT parameters: steps={chot_steps}, lr={chot_lr}")
                
                try:
                    print("DEBUG - Creating ptuning parameters")
                    # 创建正确维度的可学习参数张量 (无需梯度)
                    self.ptuning_params = torch.zeros(
                        1, self.model.lm_head.embedding_dim, 
                        device=hidden_or_intermediate_states.device,
                        dtype=hidden_or_intermediate_states.dtype
                    )
                    print(f"DEBUG - ptuning_params created: {self.ptuning_params.shape}, {self.ptuning_params.dtype}")
                    
                    # 使用简单的sign SGD
                    weight_decay = 1e-5  # 权重衰减率
                    
                    # 运行优化步骤
                    for step in range(chot_steps):
                        print(f"\nDEBUG - Optimization step {step+1}/{chot_steps}")
                        # 保存原始hidden_states副本
                        hidden_states_orig = hidden_or_intermediate_states.clone()
                        print(f"DEBUG - hidden_states_orig: {hidden_states_orig.shape}, {hidden_states_orig.dtype}")
                        
                        # 应用ptuning参数
                        patched_hidden_states = hidden_states_orig + self.ptuning_params
                        print(f"DEBUG - patched_hidden_states: {patched_hidden_states.shape}, {patched_hidden_states.dtype}")
                        
                        # 前向传播计算logits
                        print("DEBUG - Computing logits for loss calculation")
                        try:
                            temp_logits = self.model.compute_logits(patched_hidden_states, None)
                        except Exception as e:
                            print(f"DEBUG - Error in compute_logits: {e}")
                            raise
                        
                        print(f"DEBUG - temp_logits: {temp_logits.shape}, {temp_logits.dtype}")
                        
                        # 准备计算交叉熵损失
                        target_tokens = model_input.input_tokens
                        print(f"DEBUG - target_tokens: {target_tokens.shape}, {target_tokens.dtype}")
                        
                        # 根据实际输出调整形状处理
                        # 检查logits的维度
                        print(f"DEBUG - Checking logits dimensions: {len(temp_logits.shape)}")
                        if len(temp_logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                            print("DEBUG - Processing 3D logits")
                            shift_logits = temp_logits[..., :-1, :].contiguous()
                            shift_labels = target_tokens[..., 1:].contiguous()
                            batch_size = shift_logits.size(0)
                            seq_length = shift_logits.size(1)
                            vocab_size = shift_logits.size(2)
                        elif len(temp_logits.shape) == 2:  # [seq_len, vocab_size]
                            print("DEBUG - Processing 2D logits")
                            # 已经是展平的形式
                            seq_len = temp_logits.size(0)
                            vocab_size = temp_logits.size(1)
                            
                            # 调整为能够匹配的尺寸
                            shift_logits = temp_logits[:-1, :].contiguous()
                            shift_labels = target_tokens[1:].contiguous()
                            
                            # 扁平化处理 (没有batch维度)
                            batch_size = 1
                            seq_length = shift_logits.size(0)
                        else:
                            print(f"DEBUG - Unexpected logits shape: {temp_logits.shape}")
                        
                        print(f"DEBUG - shift_logits: {shift_logits.shape}, shift_labels: {shift_labels.shape}")
                        print(f"DEBUG - batch_size: {batch_size}, seq_length: {seq_length}, vocab_size: {vocab_size}")
                        
                        # 将logits转换为概率分布
                        # 计算softmax: exp(x_i) / sum(exp(x_j))
                        try:
                            print("DEBUG - Computing softmax")
                            max_logits = torch.max(shift_logits, dim=-1, keepdim=True)[0]
                            exp_logits = torch.exp(shift_logits - max_logits)
                            sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
                            probs = exp_logits / sum_exp_logits
                            print(f"DEBUG - probs: {probs.shape}, {probs.dtype}")
                        except Exception as e:
                            print(f"DEBUG - Error in softmax computation: {e}")
                            raise
                        
                        # 获取每个位置对应标签的概率 - 向量化操作
                        # 处理2D logits的情况
                        try:
                            print("DEBUG - Preparing flat tensors")
                            if len(temp_logits.shape) == 2:
                                flat_probs = probs  # 已经是扁平的形式 [seq_len, vocab_size]
                                flat_labels = shift_labels  # [seq_len]
                            else:
                                flat_probs = probs.view(-1, vocab_size)
                                flat_labels = shift_labels.view(-1)
                            print(f"DEBUG - flat_probs: {flat_probs.shape}, flat_labels: {flat_labels.shape}")
                        except Exception as e:
                            print(f"DEBUG - Error in tensor flattening: {e}")
                            raise
                        
                        # === DEBUG INFO START ===
                        print("DEBUG - Shapes:")
                        print(f"temp_logits.shape: {temp_logits.shape}, dtype: {temp_logits.dtype}")
                        print(f"shift_logits.shape: {shift_logits.shape}, dtype: {shift_logits.dtype}")
                        print(f"target_tokens.shape: {target_tokens.shape}, dtype: {target_tokens.dtype}")
                        print(f"shift_labels.shape: {shift_labels.shape}, dtype: {shift_labels.dtype}")
                        print(f"flat_probs.shape: {flat_probs.shape}, dtype: {flat_probs.dtype}")
                        print(f"flat_labels.shape: {flat_labels.shape}, dtype: {flat_labels.dtype}")
                        # === DEBUG INFO END ===
                        
                        valid_label_mask = (flat_labels >= 0).float()
                        
                        # 获取每个位置对应标签的概率 - 向量化操作
                        try:
                            print("DEBUG - Computing label probabilities")
                            # 创建索引张量 (batch_idx, label_idx)
                            batch_indices = torch.arange(flat_labels.size(0), 
                                                        device=flat_labels.device)
                            # 使用高级索引直接获取所有标签概率
                            valid_labels = torch.clamp(flat_labels, min=0)  # 将所有负索引变为0，避免索引错误
                            label_probs = flat_probs[batch_indices, valid_labels]
                            print(f"DEBUG - label_probs: {label_probs.shape}, {label_probs.dtype}")
                            
                            # 对无效标签位置应用掩码
                            masked_probs = label_probs * valid_label_mask
                            print(f"DEBUG - masked_probs: {masked_probs.shape}, {masked_probs.dtype}")
                            
                            # 计算交叉熵损失: -log(p)
                            log_probs = torch.log(masked_probs + 1e-10)
                            num_valid = torch.sum(valid_label_mask)
                            loss = -torch.sum(log_probs) / (num_valid + 1e-10)
                            print(f"DEBUG - loss: {loss.item()}")
                        except Exception as e:
                            print(f"DEBUG - Error in loss computation: {e}")
                            raise
                        
                        # === DEBUG INFO FOR LOSS ===
                        print("DEBUG - Loss calculation:")
                        print(f"masked_probs stats: min={masked_probs.min().item()}, "
                            f"max={masked_probs.max().item()}, "
                            f"mean={masked_probs.mean().item()}")
                        print(f"num_valid: {num_valid.item()}")
                        print(f"loss: {loss.item()}")
                        # === DEBUG INFO END ==
                        
                        # 手动计算梯度 - 向量化操作
                        # 对softmax的梯度: dL/ds_i = p_i - 1(i=y)
                        d_probs = flat_probs.clone()
                        
                        # === DEBUG INFO START ===
                        print("DEBUG - Gradient calculation:")
                        print(f"valid_label_mask.shape: {valid_label_mask.shape}, dtype: {valid_label_mask.dtype}")
                        print(f"valid_labels.shape: {valid_labels.shape}, dtype: {valid_labels.dtype}")
                        print(f"d_probs.shape: {d_probs.shape}, dtype: {d_probs.dtype}")
                        # === DEBUG INFO END ===
                        
                        # 创建one-hot张量，表示正确标签的位置
                        valid_labels = torch.clamp(flat_labels, min=0)  # 将所有负索引变为0
                        
                        # 确保数据类型匹配
                        one_hot = torch.zeros_like(flat_probs)
                        # 将valid_label_mask转换为与one_hot相同的数据类型
                        valid_label_mask_cast = valid_label_mask.to(dtype=one_hot.dtype)
                        
                        # === DEBUG INFO BEFORE SCATTER ===
                        print("DEBUG - Before scatter:")
                        print(f"one_hot.shape: {one_hot.shape}, dtype: {one_hot.dtype}")
                        print(f"valid_labels.unsqueeze(1).shape: {valid_labels.unsqueeze(1).shape}, "
                            f"dtype: {valid_labels.unsqueeze(1).dtype}")
                        print(f"valid_label_mask_cast.unsqueeze(1).shape: {valid_label_mask_cast.unsqueeze(1).shape}, "
                            f"dtype: {valid_label_mask_cast.unsqueeze(1).dtype}")
                        # === DEBUG INFO END ===
                        
                        try:
                            print("DEBUG - Attempting scatter operation")
                            one_hot.scatter_(1, valid_labels.unsqueeze(1), valid_label_mask_cast.unsqueeze(1))
                            print("DEBUG - Scatter operation successful")
                        except Exception as e:
                            print(f"DEBUG - Scatter error: {e}")
                            # 备选方案：手动创建one-hot
                            print("DEBUG - Using manual one-hot creation as fallback")
                            one_hot = torch.zeros_like(flat_probs)
                            for i in range(valid_labels.size(0)):
                                if valid_label_mask[i] > 0:
                                    one_hot[i, valid_labels[i]] = 1.0
                            print("DEBUG - Manual one-hot creation completed")
                        
                        # 从概率分布中减去one-hot张量
                        try:
                            print("DEBUG - Calculating final gradient")
                            d_probs = d_probs - one_hot
                            
                            # 调整梯度的缩放
                            d_probs = d_probs / (torch.sum(valid_label_mask) + 1e-10)
                            print(f"DEBUG - d_probs after scaling: {d_probs.shape}")
                        except Exception as e:
                            print(f"DEBUG - Error in gradient calculation: {e}")
                            raise
                        
                        # 反向传播到patched_hidden_states
                        try:
                            print("DEBUG - Backpropagating to hidden states")
                            # 简化：假设logits = W * hidden_states + b
                            # 则 dL/d_hidden = (dL/d_logits) * W^T
                            lm_head_weight = self.model.lm_head.weight  # [vocab_size, hidden_dim]
                            print(f"DEBUG - lm_head_weight: {lm_head_weight.shape}, {lm_head_weight.dtype}")
                            
                            # 调整d_probs的形状以适应不同情况
                            if len(temp_logits.shape) == 2:
                                # 对于2D logits，不需要重新调整形状
                                d_hidden = torch.matmul(d_probs, lm_head_weight)
                            else:
                                # 对于3D logits，先恢复batch维度
                                d_hidden = torch.matmul(d_probs.view(batch_size, seq_length, vocab_size), 
                                                    lm_head_weight)
                            print(f"DEBUG - d_hidden: {d_hidden.shape}, {d_hidden.dtype}")
                        except Exception as e:
                            print(f"DEBUG - Error in backpropagation: {e}")
                            raise
                        
                        # 梯度只影响ptuning_params (等于所有位置的梯度和)
                        try:
                            print("DEBUG - Computing gradient for ptuning_params")
                            d_ptuning = d_hidden.sum(dim=(0, 1), keepdim=True)
                            print(f"DEBUG - d_ptuning shape: {d_ptuning.shape}, dtype: {d_ptuning.dtype}")
                        except Exception as e:
                            print(f"DEBUG - Error in gradient aggregation: {e}")
                            if len(d_hidden.shape) == 2:
                                print("DEBUG - Attempting alternative aggregation for 2D tensor")
                                d_ptuning = d_hidden.sum(dim=0, keepdim=True).unsqueeze(0)
                            else:
                                raise
                        
                        # === DEBUG INFO FOR GRADIENT ===
                        print("DEBUG - Gradient and Update:")
                        print(f"d_ptuning stats: min={d_ptuning.min().item()}, "
                            f"max={d_ptuning.max().item()}, "
                            f"mean={d_ptuning.mean().item()}, "
                            f"shape={d_ptuning.shape}, "
                            f"dtype={d_ptuning.dtype}")
                        # === DEBUG INFO END ===
                        
                        # Sign SGD更新逻辑 - 只使用梯度的符号
                        try:
                            print("DEBUG - Applying Sign SGD update")
                            grad_sign = torch.sign(d_ptuning)
                            print(f"DEBUG - grad_sign: {grad_sign.shape}, {grad_sign.dtype}")
                            
                            # 更新参数
                            self.ptuning_params = self.ptuning_params - chot_lr * (
                                grad_sign + weight_decay * self.ptuning_params
                            )
                            print(f"DEBUG - Updated ptuning_params: min={self.ptuning_params.min().item()}, "
                                f"max={self.ptuning_params.max().item()}, "
                                f"mean={self.ptuning_params.mean().item()}")
                        except Exception as e:
                            print(f"DEBUG - Error in parameter update: {e}")
                            raise
                    
                except Exception as e:
                    print(f"DEBUG - Exception in CHOT optimization: {e}")
                    traceback.print_exc()
                finally:
                    # 清理
                    print("DEBUG - Cleaning up")
                    torch.cuda.empty_cache()
                    
                    # 设置环境变量标记已优化
                    os.environ["CHOT_OPTIMIZED"] = "1"
                    print("DEBUG - CHOT optimization completed")
            
            # 应用优化后的参数
            if hasattr(self, 'ptuning_params') and self.ptuning_params is not None:
                print("DEBUG - Applying ptuning parameters to hidden states")
                print(f"DEBUG - Before: hidden_states shape: {hidden_or_intermediate_states.shape}")
                hidden_or_intermediate_states = hidden_or_intermediate_states + self.ptuning_params
                print(f"DEBUG - After: hidden_states shape: {hidden_or_intermediate_states.shape}")
        

