- 라벨 : 2개

- DataSets (MIntRec)
    - Train : 1334 개
    - Valid(Test) : 445 개

- 학습 결과(== 문제점)
    - 학습이 제대로 진행이 안됨. Loss도 안떨어지고, Acc도 상승안함.
    - Loss : 1e-3 ~ 5e-5 다양하게 시도했지만, 학습 안됨.

- 학습 시간
    - 4060 Ti : 1ep 당 20초 미만

- 학습 로그 

```
# Train Dataset Log
{'loss': 0.6902, 'learning_rate': 4.8333333333333334e-05, 'epoch': 1.0}
{'loss': 0.6877, 'learning_rate': 4.666666666666667e-05, 'epoch': 2.0}
{'loss': 0.6874, 'learning_rate': 4.5e-05, 'epoch': 3.0}
{'loss': 0.6872, 'learning_rate': 4.3333333333333334e-05, 'epoch': 4.0}
{'loss': 0.6873, 'learning_rate': 4.166666666666667e-05, 'epoch': 5.0}
{'loss': 0.6892, 'learning_rate': 4e-05, 'epoch': 6.0}
{'loss': 0.687, 'learning_rate': 3.8333333333333334e-05, 'epoch': 7.0}
{'loss': 0.687, 'learning_rate': 3.6666666666666666e-05, 'epoch': 8.0}
{'loss': 0.6867, 'learning_rate': 3.5e-05, 'epoch': 9.0}
{'loss': 0.6861, 'learning_rate': 3.3333333333333335e-05, 'epoch': 10.0}
{'loss': 0.6875, 'learning_rate': 3.1666666666666666e-05, 'epoch': 11.0}
{'loss': 0.6871, 'learning_rate': 3e-05, 'epoch': 12.0}
{'loss': 0.6861, 'learning_rate': 2.8333333333333335e-05, 'epoch': 13.0}
{'loss': 0.6865, 'learning_rate': 2.6666666666666667e-05, 'epoch': 14.0}
{'loss': 0.6867, 'learning_rate': 2.5e-05, 'epoch': 15.0}
{'loss': 0.6859, 'learning_rate': 2.3333333333333336e-05, 'epoch': 16.0}
{'loss': 0.6861, 'learning_rate': 2.1666666666666667e-05, 'epoch': 17.0}
{'loss': 0.6859, 'learning_rate': 2e-05, 'epoch': 18.0}
{'loss': 0.6862, 'learning_rate': 1.8333333333333333e-05, 'epoch': 19.0}
{'loss': 0.6867, 'learning_rate': 1.6666666666666667e-05, 'epoch': 20.0}
{'loss': 0.6862, 'learning_rate': 1.5e-05, 'epoch': 21.0}
{'loss': 0.686, 'learning_rate': 1.3333333333333333e-05, 'epoch': 22.0}
{'loss': 0.6859, 'learning_rate': 1.1666666666666668e-05, 'epoch': 23.0}
{'loss': 0.6856, 'learning_rate': 1e-05, 'epoch': 24.0}
{'loss': 0.6856, 'learning_rate': 8.333333333333334e-06, 'epoch': 25.0}
{'loss': 0.6854, 'learning_rate': 6.666666666666667e-06, 'epoch': 26.0}
{'loss': 0.6861, 'learning_rate': 5e-06, 'epoch': 27.0}
{'loss': 0.6857, 'learning_rate': 3.3333333333333333e-06, 'epoch': 28.0}
{'loss': 0.6856, 'learning_rate': 1.6666666666666667e-06, 'epoch': 29.0}
{'loss': 0.6857, 'learning_rate': 0.0, 'epoch': 30.0}

# Valid Dataset Log
{'eval_loss': 0.6882697939872742, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8809, 'eval_samples_per_second': 505.154, 'eval_steps_per_second': 63.57, 'epoch': 1.0}
{'eval_loss': 0.6865996718406677, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8613, 'eval_samples_per_second': 516.66, 'eval_steps_per_second': 65.018, 'epoch': 2.0}
{'eval_loss': 0.6871669888496399, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8619, 'eval_samples_per_second': 516.287, 'eval_steps_per_second': 64.971, 'epoch': 3.0}
{'eval_loss': 0.6865999698638916, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8828, 'eval_samples_per_second': 504.068, 'eval_steps_per_second': 63.433, 'epoch': 4.0}
{'eval_loss': 0.6867841482162476, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8717, 'eval_samples_per_second': 510.469, 'eval_steps_per_second': 64.239, 'epoch': 5.0}
{'eval_loss': 0.6866423487663269, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8654, 'eval_samples_per_second': 514.24, 'eval_steps_per_second': 64.713, 'epoch': 6.0}
{'eval_loss': 0.686898410320282, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.865, 'eval_samples_per_second': 514.452, 'eval_steps_per_second': 64.74, 'epoch': 7.0}
{'eval_loss': 0.6865676045417786, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8614, 'eval_samples_per_second': 516.607, 'eval_steps_per_second': 65.011, 'epoch': 8.0}
{'eval_loss': 0.68658846616745, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.864, 'eval_samples_per_second': 515.018, 'eval_steps_per_second': 64.811, 'epoch': 9.0}
{'eval_loss': 0.6873876452445984, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.863, 'eval_samples_per_second': 515.637, 'eval_steps_per_second': 64.889, 'epoch': 10.0}
{'eval_loss': 0.6866733431816101, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8606, 'eval_samples_per_second': 517.059, 'eval_steps_per_second': 65.068, 'epoch': 11.0}
{'eval_loss': 0.6866382360458374, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8598, 'eval_samples_per_second': 517.551, 'eval_steps_per_second': 65.13, 'epoch': 12.0}
{'eval_loss': 0.6867321729660034, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8611, 'eval_samples_per_second': 516.809, 'eval_steps_per_second': 65.037, 'epoch': 13.0}
{'eval_loss': 0.6868332028388977, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8602, 'eval_samples_per_second': 517.303, 'eval_steps_per_second': 65.099, 'epoch': 14.0}
{'eval_loss': 0.6869437098503113, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8586, 'eval_samples_per_second': 518.274, 'eval_steps_per_second': 65.221, 'epoch': 15.0}
{'eval_loss': 0.6865677237510681, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8616, 'eval_samples_per_second': 516.479, 'eval_steps_per_second': 64.995, 'epoch': 16.0}
{'eval_loss': 0.6865809559822083, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8612, 'eval_samples_per_second': 516.729, 'eval_steps_per_second': 65.027, 'epoch': 17.0}
{'eval_loss': 0.6865950226783752, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8501, 'eval_samples_per_second': 523.497, 'eval_steps_per_second': 65.878, 'epoch': 18.0}
{'eval_loss': 0.6865891218185425, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8722, 'eval_samples_per_second': 510.222, 'eval_steps_per_second': 64.208, 'epoch': 19.0}
{'eval_loss': 0.6867563128471375, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8602, 'eval_samples_per_second': 517.327, 'eval_steps_per_second': 65.102, 'epoch': 20.0}
{'eval_loss': 0.6866738796234131, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8595, 'eval_samples_per_second': 517.736, 'eval_steps_per_second': 65.153, 'epoch': 21.0}
{'eval_loss': 0.686581552028656, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8612, 'eval_samples_per_second': 516.751, 'eval_steps_per_second': 65.029, 'epoch': 22.0}
{'eval_loss': 0.6866324543952942, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8598, 'eval_samples_per_second': 517.568, 'eval_steps_per_second': 65.132, 'epoch': 23.0}
{'eval_loss': 0.6866426467895508, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8589, 'eval_samples_per_second': 518.082, 'eval_steps_per_second': 65.197, 'epoch': 24.0}
{'eval_loss': 0.6866879463195801, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8627, 'eval_samples_per_second': 515.83, 'eval_steps_per_second': 64.913, 'epoch': 25.0}
{'eval_loss': 0.6866068243980408, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8584, 'eval_samples_per_second': 518.399, 'eval_steps_per_second': 65.237, 'epoch': 26.0}
{'eval_loss': 0.6866294145584106, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.861, 'eval_samples_per_second': 516.833, 'eval_steps_per_second': 65.04, 'epoch': 27.0}
{'eval_loss': 0.6866488456726074, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8583, 'eval_samples_per_second': 518.444, 'eval_steps_per_second': 65.242, 'epoch': 28.0}
{'eval_loss': 0.6866346597671509, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8605, 'eval_samples_per_second': 517.125, 'eval_steps_per_second': 65.076, 'epoch': 29.0}
{'eval_loss': 0.6866279244422913, 'eval_accuracy': 0.5573033707865168, 'eval_f1': 0.5573033707865168, 'eval_precision': 0.5573033707865168, 'eval_recall': 0.5573033707865168, 'eval_runtime': 0.8611, 'eval_samples_per_second': 516.754, 'eval_steps_per_second': 65.03, 'epoch': 30.0}
{'train_runtime': 417.0356, 'train_samples_per_second': 95.963, 'train_steps_per_second': 12.013, 'train_loss': 0.6866049639003243, 'epoch': 30.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5010/5010 [06:57<00:00, 12.01it/s]
```