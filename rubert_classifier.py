import sys
import os
import time
import torch
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


@dataclass
class ClassificationResult:
    request: str
    probability: float
    is_blocked: int


class RuBERTClassifier:
    def __init__(self, model_name: str = "cointegrated/rubert-tiny-toxicity", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().tolist()

        prob_harmful = 1.0 - probs[0] * (1.0 - probs[4])
        return float(prob_harmful)

    def predict(self, text: str) -> int:
        prob = self.predict_proba(text)
        return 1 if prob > self.threshold else 0

    def check_request(self, text: str) -> ClassificationResult:
        prob = self.predict_proba(text)
        is_blocked = 1 if prob > self.threshold else 0
        return ClassificationResult(
            request=text,
            probability=prob,
            is_blocked=is_blocked
        )

    def filter_dataframe(self, df: pd.DataFrame, column: str = 'request') -> pd.DataFrame:
        results = df[column].apply(self.check_request)
        df_out = df.copy()
        df_out['toxicity_prob'] = results.apply(lambda r: round(r.probability, 4))
        df_out['prediction'] = results.apply(lambda r: r.is_blocked)
        return df_out


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total_pos = tp + fn
    total_neg = tn + fp

    tpr = tp / total_pos if total_pos else 0.0
    fpr = fp / total_neg if total_neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
    asr = 1.0 - tpr

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'TPR': tpr, 'FPR': fpr, 'Precision': precision, 'F1': f1, 'ASR': asr
    }


def main():
    dataset_file = "dataset_utf8.txt"
    if not os.path.exists(dataset_file):
        print(f"Ошибка: файл {dataset_file} не найден.")
        return

    df = pd.read_csv(dataset_file, encoding='utf-8')
    df.columns = ['request', 'category', 'label']
    df['label'] = df['label'].astype(int)

    print("Загрузка модели RuBERT (cointegrated/rubert-tiny-toxicity)...")
    classifier = RuBERTClassifier(threshold=0.5)
    print("Модель успешно инициализирована.\n")

    start_time = time.perf_counter()
    df_result = classifier.filter_dataframe(df, column='request')
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    latency_ms = total_time_ms / len(df)
    throughput = len(df) / (end_time - start_time)

    metrics = calculate_metrics(df_result['label'], df_result['prediction'])

    print("=== Результаты работы RuBERT-классификатора ===")
    print(f"Порог отсечения (threshold): {classifier.threshold}")
    print(f"Запросов проверено:         {len(df)}")
    print(f"Матрица ошибок:             TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"Защищенность (TPR / Recall): {metrics['TPR']*100:.1f}%")
    print(f"Ложные срабатывания (FPR):    {metrics['FPR']*100:.1f}%")
    print(f"Точность (Precision):         {metrics['Precision']*100:.1f}%")
    print(f"F1-мера:                      {metrics['F1']:.4f}")
    print(f"Успешность атак (ASR):        {metrics['ASR']*100:.1f}%")
    print(f"Средняя задержка (Latency):   {latency_ms:.2f} мс/запрос")
    print(f"Пропускная способность:       {throughput:.1f} запросов/сек")

    export_df = df_result[['request', 'category', 'label', 'toxicity_prob', 'prediction']].copy()
    output_csv = "rubert_classifier_results.csv"
    export_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nПодробные результаты сохранены в {output_csv}")


if __name__ == '__main__':
    main()
