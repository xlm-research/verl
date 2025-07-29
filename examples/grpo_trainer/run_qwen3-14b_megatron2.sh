set -x

# export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

rollout_mode="async"
if [ "$rollout_mode" = "async" ]; then
   export VLLM_USE_V1=1
   return_raw_chat="True"
fi


train_files=/mnt/lustre/share_data/datasets/gsm8k/train.parquet
test_files=/mnt/lustre/share_data/datasets/gsm8k/test.parquet

USE_FUSED_KERNELS=True
OFF_LOAD=True

TP=4
PP=1
GEN_TP=4


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/lustre/share_data/models/Qwen/Qwen3-14B \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=/mnt/lustre/yangbo1/verl/models/Qwen3-14B_dist_ckpt_mcore \
    actor_rollout_ref.actor.megatron.param_offload=$OFF_LOAD \
    actor_rollout_ref.actor.megatron.optimizer_offload=$OFF_LOAD \
    actor_rollout_ref.actor.megatron.grad_offload=$OFF_LOAD \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen3_14b_megatron' \
    trainer.profile_steps=[4,5] \
    trainer.n_gpus_per_node=8 \
    trainer.resume_mode=False \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
