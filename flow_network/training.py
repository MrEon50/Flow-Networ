import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .models import FlowNetwork, EnhancedFlowTransformer
from .utils import safe_tensor_to_int

class MultiTaskFlowLoss(nn.Module):
    """
    Advanced Multi-Task Loss Function for LLM training
    Includes context consistency, coherence, and conversational losses
    """

    def __init__(self, diversity_weight: float = 0.001,
                 context_weight: float = 0.1,
                 coherence_weight: float = 0.05,
                 conversation_weight: float = 0.08,
                 memory_weight: float = 0.02):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.context_weight = context_weight
        self.coherence_weight = coherence_weight
        self.conversation_weight = conversation_weight
        self.memory_weight = memory_weight

        # Coherence loss components
        self.coherence_criterion = nn.CosineSimilarity(dim=-1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                metrics_list: List[Dict],
                context_features: Optional[torch.Tensor] = None,
                conversation_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:

        # Main task loss (language modeling)
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # Context consistency loss
        context_loss = self._context_consistency_loss(context_features, logits) if context_features is not None else 0.0

        # Coherence loss for dialog flow
        coherence_loss = self._coherence_loss(logits)

        # Conversation continuity loss
        conversation_loss = self._conversation_loss(logits, conversation_history) if conversation_history is not None else 0.0

        # Memory efficiency loss
        memory_loss = self._memory_efficiency_loss(metrics_list)

        # Diversity regularization
        diversity_reg = self._calculate_diversity(metrics_list)

        # Total loss with adaptive weighting
        total_loss = (task_loss +
                     self.context_weight * context_loss +
                     self.coherence_weight * coherence_loss +
                     self.conversation_weight * conversation_loss +
                     self.memory_weight * memory_loss +
                     self.diversity_weight * diversity_reg)

        loss_info = {
            'total': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'task': task_loss.item(),
            'context': context_loss.item() if hasattr(context_loss, 'item') else context_loss,
            'coherence': coherence_loss.item() if hasattr(coherence_loss, 'item') else coherence_loss,
            'conversation': conversation_loss.item() if hasattr(conversation_loss, 'item') else conversation_loss,
            'memory': memory_loss.item() if hasattr(memory_loss, 'item') else memory_loss,
            'diversity': diversity_reg
        }

        return total_loss, loss_info

    def _context_consistency_loss(self, context_features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Ensure consistency between context and generated tokens"""
        if context_features is None:
            return torch.tensor(0.0, device=logits.device)

        # Calculate similarity between context and output representations
        batch_size, seq_len, vocab_size = logits.shape

        # Convert logits to embeddings (simplified)
        output_probs = F.softmax(logits, dim=-1)

        # Context consistency: adjacent tokens should have similar context influence
        context_diff = torch.diff(context_features, dim=1)
        context_consistency = torch.mean(torch.norm(context_diff, dim=-1))

        return context_consistency

    def _coherence_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Ensure coherent flow in generated sequences"""
        batch_size, seq_len, vocab_size = logits.shape

        if seq_len < 2:
            return torch.tensor(0.0, device=logits.device)

        # Calculate coherence between adjacent tokens
        probs = F.softmax(logits, dim=-1)

        # Coherence: adjacent tokens should have smooth probability transitions
        prob_diff = torch.diff(probs, dim=1)
        coherence_penalty = torch.mean(torch.norm(prob_diff, dim=-1))

        return coherence_penalty

    def _conversation_loss(self, logits: torch.Tensor, conversation_history: torch.Tensor) -> torch.Tensor:
        """Ensure conversation continuity and relevance"""
        if conversation_history is None:
            return torch.tensor(0.0, device=logits.device)

        # Simple conversation continuity: current output should be related to history
        current_probs = F.softmax(logits, dim=-1)
        history_probs = F.softmax(conversation_history, dim=-1)

        # Calculate relevance score
        relevance = self.coherence_criterion(
            current_probs.mean(dim=1),
            history_probs.mean(dim=1)
        ).mean()

        # We want high relevance, so minimize negative relevance
        return -relevance

    def _memory_efficiency_loss(self, metrics_list: List[Dict]) -> torch.Tensor:
        """Encourage efficient memory usage"""
        memory_usage_values = []

        for metrics in metrics_list:
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if 'memory_usage' in key and isinstance(value, (int, float)):
                        memory_usage_values.append(value)

        if not memory_usage_values:
            return torch.tensor(0.0)

        # Encourage moderate memory usage (not too high, not too low)
        avg_memory_usage = np.mean(memory_usage_values)
        optimal_usage = 0.7  # Target 70% memory usage

        memory_penalty = abs(avg_memory_usage - optimal_usage)
        return torch.tensor(memory_penalty, dtype=torch.float32)

    def _calculate_diversity(self, metrics_list: List[Dict]) -> float:
        """Calculate diversity regularization from metrics"""
        diversity_values = []

        for metrics in metrics_list:
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.mean().item()

                    if ('diversity' in key or 'entropy' in key) and isinstance(value, (int, float)):
                        diversity_values.append(value)

        # Diversity regularization (maximize diversity)
        return -np.mean(diversity_values) if diversity_values else 0.0


class FlowLoss(nn.Module):
    """Advanced loss function optimized for Flow Networks"""
    
    def __init__(self, diversity_weight: float = 0.001):
        super().__init__()
        self.diversity_weight = diversity_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                metrics_list: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        
        # Task loss
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Collect diversity metrics
        diversity_values = []
        
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.mean().item()
                
                if 'diversity' in key or 'entropy' in key:
                    diversity_values.append(value)
        
        # Diversity regularization (maximize)
        diversity_reg = -np.mean(diversity_values) if diversity_values else 0.0
        
        # Total loss
        total_loss = task_loss + self.diversity_weight * diversity_reg
        
        loss_info = {
            'total': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'task': task_loss.item(),
            'diversity': diversity_reg
        }
        
        return total_loss, loss_info


def train_flow_network(model: FlowNetwork, data: List, num_epochs: int = 1,
                      lr: float = 1e-3, device: str = 'cpu') -> Dict:
    """Train Flow Network with optimized settings"""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = FlowLoss(diversity_weight=0.001)

    training_metrics = {
        'losses': [],
        'times': [],
        'throughputs': []
    }

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_time = 0

        for batch_idx, (input_ids, targets) in enumerate(data):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            start_time = time.time()

            optimizer.zero_grad()
            logits, metrics = model(input_ids)
            loss, loss_info = loss_fn(logits, targets, metrics)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_time = time.time() - start_time
            batch_throughput = input_ids.numel() / batch_time

            epoch_loss += loss.item()
            epoch_time += batch_time

            training_metrics['losses'].append(loss.item())
            training_metrics['times'].append(batch_time)
            training_metrics['throughputs'].append(batch_throughput)

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(data)}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Throughput: {batch_throughput:.0f} tokens/sec")
                for key, value in loss_info.items():
                    print(f"  {key}: {value:.6f}")

        avg_loss = epoch_loss / len(data)
        avg_throughput = np.mean(training_metrics['throughputs'][-len(data):])

        print(f"Epoch {epoch+1} completed:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average throughput: {avg_throughput:.0f} tokens/sec")
        print(f"  Total time: {epoch_time:.2f}s\n")

    return training_metrics


