from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, TrainingArguments, Trainer
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import pandas as pd 
import os 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class AudioDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = item['labels'].long()  # 레이블을 LongTensor로 변환
        return item

    def __len__(self):
        return len(self.encodings.input_values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# tsv 파일 불러오기 
train = pd.read_csv(r'data\train.tsv', sep='\t')
test = pd.read_csv(r'data\test.tsv', sep='\t')
dev = pd.read_csv(r'data\dev.tsv', sep='\t')

# 세부 카테고리 (20개)
# label_to_index = {
#     'Inform' : 0 ,
#     'Complain' : 1, 
#     'Praise' : 2, 
#     'Apologise': 3,
#     'Thank' : 4,
#     'Criticize': 5,
#     'Advise': 6,
#     'Arrange': 7,
#     'Introduce': 8,
#     'Care': 9,
#     'Comfort' : 10,
#     'Leave' : 11,
#     'Prevent' : 12,
#     'Taunt' : 13,
#     'Greet' : 14,
#     'Agree' : 15,
#     'Ask for help' : 16,
#     'Joke' : 17,
#     'Oppose' : 18,
#     'Flaunt' : 19
# }
# 대 카테고리 (2개)
label_to_index = {
    'Inform' : 1 ,
    'Complain' : 0, 
    'Praise' : 0, 
    'Apologise': 0,
    'Thank' : 0,
    'Criticize': 0,
    'Advise': 1,
    'Arrange': 1,
    'Introduce': 1,
    'Care': 0,
    'Comfort' : 1,
    'Leave' : 1,
    'Prevent' : 1,
    'Taunt' : 0,
    'Greet' : 1,
    'Agree' : 0,
    'Ask for help' : 1,
    'Joke' : 0,
    'Oppose' : 0,
    'Flaunt' : 0
}
index_to_label = {v : k for k,v in label_to_index.items()}

# 라벨 개수 
num_labels = len(set(label_to_index.values()))
print('Label 개수 :', num_labels)

# Train, Test 오디오 및 라벨 
train_audios = []
train_labels = []

test_audios = []
test_labels = []

# 오디오 로드 (Train)
for row in train.values:
    # row : ['S05' 'E15' 248 "that's a really good point." 'Praise']
    wav_file_path = os.path.join('data', 'raw_audio', row[0], row[1], f'{row[2]}.wav')
    
    audio_file, sr = sf.read(wav_file_path)
    train_audios.append(audio_file)
    train_labels.append(label_to_index[row[-1]])
# 오디오 로드 (Test)
for row in test.values:
    # row : ['S05' 'E15' 248 "that's a really good point." 'Praise']
    wav_file_path = os.path.join('data', 'raw_audio', row[0], row[1], f'{row[2]}.wav')
    
    audio_file, sr = sf.read(wav_file_path)
    test_audios.append(audio_file)
    test_labels.append(label_to_index[row[-1]])

train_labels = np.array(train_labels) # 각 오디오 데이터에 해당하는 레이블
test_labels = np.array(test_labels) # 각 오디오 데이터에 해당하는 레이블


# 데이터를 훈련 데이터셋과 검증 데이터셋으로 분리
# train_data, val_data, train_labels, val_labels = train_test_split(train_audios, train_labels, test_size=0.2, random_state=42)
train_data = train_audios
train_labels = train_labels
val_data = test_audios
val_labels = test_labels

print('Train 개수 :',len(train_data))
print('Valid 개수 :',len(val_data))
print({x : list(train_labels).count(x) for x in set(train_labels)})
print({x : list(val_labels).count(x) for x in set(val_labels)})


# 모델과 프로세서 로딩
# facebook/wav2vec2-base-960h
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=num_labels).to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# 훈련 데이터를 모델에 맞게 전처리
train_input_values = processor(train_data, sampling_rate=16_000, return_tensors="pt", padding=True, truncation=True, max_length=16000)
# train_input_values = feature_extractor(raw_speech = train_data, sampling_rate=16_000, return_tensors="pt", padding=True, truncation=True, max_length=16000)
train_input_values["labels"] = torch.tensor(train_labels)

# 검증 데이터를 모델에 맞게 전처리
val_input_values = processor(val_data, sampling_rate=16_000, return_tensors="pt", padding=True, truncation=True, max_length=16000)
# val_input_values = feature_extractor(raw_speech = val_data, sampling_rate=16_000, return_tensors="pt", padding=True, truncation=True, max_length=16000)
val_input_values["labels"] = torch.tensor(val_labels)

# 데이터를 Dataset 객체로 변환
train_dataset = AudioDataset(train_input_values)
val_dataset = AudioDataset(val_input_values)



# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",          # 출력 디렉토리. 기본값: "./results"
    num_train_epochs=30,            # 총 훈련 에폭 수. 기본값: 3
    per_device_train_batch_size=8,  # 각 장치 당 훈련 배치 크기. 기본값: 8
    per_device_eval_batch_size=8,   # 각 장치 당 평가 배치 크기. 기본값: 8
    #warmup_steps=None,               # 웜업 스텝 수. 기본값: 0
    #weight_decay=0.0,                # 가중치 감소. 기본값: 0.0
    logging_dir='./logs',            # 로그 디렉토리. 기본값: "runs/{datetime}"
    logging_strategy="epoch",        # 로깅 전략 ("steps" or "epoch"). 기본값: "steps"
    logging_steps=100,               # 로깅 스텝 수. 기본값: 500
    evaluation_strategy="epoch",     # 평가 전략 ("steps", "epoch", or "no"). 기본값: "steps"
    #eval_steps=None,                 # 평가 스텝 수. 기본값: None, evaluation_strategy="steps"일 때는 logging_steps와 같음
    load_best_model_at_end=False,    # 훈련이 끝날 때 가장 좋은 모델을 불러올지 여부. 기본값: False
    metric_for_best_model='accuracy',      # 가장 좋은 모델을 선택하는 기준이 될 메트릭. 기본값: None
    #greater_is_better=None,          # metric_for_best_model이 클수록 좋은지 여부. 기본값: None, metric_for_best_model이 설정된 경우 자동으로 결정됨
    save_strategy="epoch",           # 모델 저장 전략 ("steps", "epoch", or "no"). 기본값: "steps"
    #save_steps=None,                 # 모델 저장 스텝 수. 기본값: None, save_strategy="steps"일 때는 logging_steps와 같음
    save_total_limit=5,           # 저장할 체크포인트의 최대 개수. 기본값: None
    learning_rate=5e-5,              # 학습률. 기본값: 5e-5
    lr_scheduler_type='linear'       # 학습률 스케줄러 유형 ("linear", "cosine", "cosine_with_restarts", "polynomial", or "constant"). 기본값: "linear"
)

# 훈련을 위한 Trainer 설정
trainer = Trainer(
    model=model,                         # 파인 튜닝할 모델
    args=training_args,                  # 훈련 설정
    train_dataset=train_dataset,         # 훈련 데이터셋
    eval_dataset=val_dataset,            # 검증 데이터셋
    compute_metrics=compute_metrics,     # 에폭마다 평가를 위한 함수
)

# 훈련 시작
trainer.train()