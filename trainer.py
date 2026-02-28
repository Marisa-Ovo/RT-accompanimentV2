"""
Transformer模型训练器
支持分布式训练、混合精度、梯度累积等特性
训练日志会保存在TensorBoard中，使用命令查看: tensorboard --logdir=<log_dir>
"""
import os
import torch
from datetime import datetime
from typing import Optional
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TransformerTrainer:
    """Transformer模型训练器

    封装了模型训练的完整流程，包括：
    - 优化器和学习率调度器初始化
    - 分布式训练支持
    - 训练和评估循环
    - 模型检查点保存
    - TensorBoard日志记录
    """

    def __init__(self, config, model, train_dataloader, test_dataloader=None):
        """初始化训练器

        Args:
            config: 训练配置对象，包含学习率、训练轮数等参数
            model: 待训练的PyTorch模型
            train_dataloader: 训练数据加载器
            test_dataloader: 测试数据加载器（可选）
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.global_step = 0

        # 初始化优化器和学习率调度器
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

        # 初始化Accelerator进行分布式训练
        self.accelerator = self._initialize_accelerator()

        # 准备训练组件（自动处理分布式、混合精度等）
        self._prepare_training_components()

        # 初始化TensorBoard日志
        self.accelerator.init_trackers(self.config.tensorboard_log_name)

        # 计算测试频率
        self.test_interval_steps = self._calculate_test_interval()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.99),  # LLaMA推荐的beta值
            weight_decay=0.1,
        )

    def _setup_lr_scheduler(self):
        """配置学习率调度器"""
        num_training_steps = int(
            (len(self.train_dataloader) * self.config.num_epochs * 1.5)
            / self.config.gradient_accumulation_steps
        )
        return get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _initialize_accelerator(self) -> Accelerator:
        """初始化Accelerator用于分布式训练和混合精度"""
        ddp_kwargs = DistributedDataParallelKwargs()
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=self.config.tensorboard_log_dir,
            device_placement=True,
            kwargs_handlers=[ddp_kwargs]
        )

        # 创建日志目录
        if accelerator.is_main_process and self.config.tensorboard_log_dir:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

        return accelerator

    def _prepare_training_components(self):
        """准备训练组件（处理分布式、混合精度等）"""
        components = [self.model, self.optimizer, self.train_dataloader, self.lr_scheduler]

        if self.test_dataloader is not None:
            components.insert(3, self.test_dataloader)
            prepared = self.accelerator.prepare(*components)
            self.model, self.optimizer, self.train_dataloader, self.test_dataloader, self.lr_scheduler = prepared
        else:
            prepared = self.accelerator.prepare(*components)
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = prepared

    def _calculate_test_interval(self) -> Optional[int]:
        """计算测试评估的间隔步数"""
        if self.config.use_test_set and self.test_dataloader is not None:
            interval = int(len(self.train_dataloader) * self.config.test_frequency)
            print(f"测试间隔: 每 {interval} 步测试一次 (约{self.config.test_frequency}个epoch)")
            return interval
        return None

    def train(self):
        """执行完整的训练流程"""
        for epoch in range(self.config.num_epochs):
            self._train_one_epoch(epoch)
            self._save_checkpoint_if_needed(epoch)

    def _train_one_epoch(self, epoch: int):
        """训练一个epoch

        Args:
            epoch: 当前训练轮数
        """
        self.model.train()
        progress_bar = tqdm(total=len(self.train_dataloader), disable= not self.config.log)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_start_step = self.global_step 

        for batch in self.train_dataloader:
            # 前向传播和反向传播
            loss = self._training_step(batch)
            
            # 记录日志
            if self.global_step % 10 == 0:
                self._log_training_metrics(loss, progress_bar)

            # 定期测试评估
            if self._should_evaluate(epoch_start_step):
                self._evaluate_test()

            # 定期保存检查点
            if self._should_save_checkpoint():
                self._save_checkpoint(f"steps_{self.global_step}")

            # 更新进度条
            if self.config.log:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    lr=self.lr_scheduler.get_last_lr()[0]
                )
            self.global_step += 1

    def _training_step(self, batch) -> torch.Tensor:
        """执行一个训练步骤

        Args:
            batch: 输入的批次数据

        Returns:
            当前批次的损失值
        """
        with self.accelerator.accumulate(self.model):
            # 前向传播
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"]
            )
            loss = outputs.loss

            # 反向传播
            self.accelerator.backward(loss)

            # 梯度裁剪
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            # 优化器步进
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss

    def _log_training_metrics(self, loss: torch.Tensor, progress_bar: tqdm):
        """记录训练指标到TensorBoard和进度条"""
        metrics = {
            "train/loss": loss.detach().item(),
            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "train/step": self.global_step,
        }

        # 记录到TensorBoard
        self.accelerator.log(metrics, step=self.global_step)

    def _should_evaluate(self, epoch_start_step: int) -> bool:
        """判断是否需要进行测试评估"""
        if self.test_interval_steps is None:
            return False

        return (
            self.global_step % self.test_interval_steps == 0
            and self.global_step > epoch_start_step
            and self.accelerator.is_main_process
        )

    def _evaluate_test(self):
        """在测试集上评估模型"""
        if self.test_dataloader is None:
            return

        self.model.eval()
        total_loss = 0
        num_batches = 0

        print(f"\n开始测试集评估 (step {self.global_step})...")

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="测试中"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"]
                )
                total_loss += outputs.loss.item()
                num_batches += 1

        # 计算平均损失和困惑度
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # 记录测试指标
        if self.config.test_save_results:
            test_metrics = {
                "test/loss": avg_loss,
                "test/perplexity": perplexity,
                "test/step": self.global_step,
            }
            self.accelerator.log(test_metrics, step=self.global_step)

        print(f"测试集 - 损失: {avg_loss:.4f}, 困惑度: {perplexity:.4f}")

        self.model.train()

    def _should_save_checkpoint(self) -> bool:
        """判断是否需要保存检查点（按步数）"""
        if self.config.save_steps is None or self.config.save_steps <= 0:
            return False

        return (
            self.global_step % self.config.save_steps == 0
            and self.accelerator.is_main_process
        )

    def _save_checkpoint_if_needed(self, epoch: int):
        """根据epoch判断是否需要保存检查点"""
        if not self.accelerator.is_main_process:
            return

        should_save = (
            (epoch + 1) % self.config.save_model_epochs == 0
            or epoch == self.config.num_epochs - 1
        )

        if should_save:
            self._save_checkpoint(f"epoch_{epoch}")

    def _save_checkpoint(self, prefix: str):
        """保存模型检查点

        Args:
            prefix: 检查点文件名前缀
        """
        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            timestamp = datetime.now().strftime("%m%d_%H%M")
            save_path = f"{self.config.output_dir}/{prefix}_{timestamp}"
            unwrapped_model.save_pretrained(save_path)
            print(f"模型已保存至: {save_path}")
