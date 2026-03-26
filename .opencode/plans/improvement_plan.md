# Improvement Plan for Text Prediction Model

## Current Issues Identified

1. **Severe Overfitting**: Training accuracy improves while validation accuracy remains at 0.0000
2. **Insufficient Data**: Only 120 conversations for 80 intents (~1.5 examples per intent)
3. **Excessive Model Complexity**: VOCAB_SIZE=700, EMBEDDING_DIM=256, MAX_LEN=600 creates too many parameters
4. **Ineffective Data Augmentation**: Simple word reversal doesn't create meaningful variations
5. **Excessive Sequence Length**: MAX_LEN=600 is unnecessarily long for chatbot inputs

## Proposed Solutions

### 1. Reduce Model Complexity
- Decrease VOCAB_SIZE from 700 to 300-400
- Reduce EMBEDDING_DIM from 256 to 128
- Lower NUM_NEURONS from 150 to 64-80
- Consider removing attention mechanism initially

### 2. Improve Dataset Utilization
- Implement better data augmentation techniques:
  - Synonym replacement using Spanish WordNet
  - Random insertion/deletion of words
  - Back-translation (Spanish -> English -> Spanish)
- Increase effective dataset size by 3-5x through augmentation

### 3. Optimize Sequence Length
- Analyze actual input lengths in dataset
- Set MAX_LEN to 95th percentile of input lengths (likely < 100)
- Current MAX_LEN=600 is wasteful and increases noise

### 4. Adjust Training Parameters
- Increase dropout rate from 0.4 to 0.5
- Further reduce L2 regularization from 1e-5 to 1e-6
- Consider using learning rate warmup for more epochs

### 5. Address Class Imbalance
- Implement class weighting in loss function
- Use stratified sampling in data preparation
- Consider focal loss for severely imbalanced classes

### 6. Simplify Architecture (Initial Approach)
- Start with simpler model: Embedding -> LSTM -> Dense
- Add complexity only if needed
- Validate each addition improves validation metrics

## Implementation Priority

1. **Immediate**: Reduce model complexity and adjust MAX_LEN
2. **Short-term**: Implement effective data augmentation
3. **Medium-term**: Address class imbalance and adjust regularization
4. **Long-term**: Experiment with architecture improvements

## Expected Outcomes

- Reduced gap between training and validation performance
- Validation accuracy > 5% (currently 0%)
- Stable or improved training accuracy
- Better generalization to unseen inputs

## Files to Modify

1. `text-model-6/model.py` - Main model architecture and hyperparameters
2. Potentially create new data augmentation script

## Success Metrics

- Validation accuracy > 5% by epoch 5
- Validation loss decreasing or stabilizing
- Meaningful responses to test inputs (not just fallback messages)