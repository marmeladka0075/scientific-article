import sys
import os
import re
import time
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field


if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


@dataclass
class FilterResult:
    request: str
    is_blocked: int
    matched_rules: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


class RegexFilter:
    def __init__(self):
        flags = re.IGNORECASE | re.UNICODE

        self.rules: Dict[str, List[re.Pattern]] = {
            "Написание вредоносного кода": [
                re.compile(r'вредоносн\w*\s+(?:код\w*|по|программ\w*|скрипт\w*|библиотек\w*)', flags),
                re.compile(r'(?:написа|созда|разработ\w*|скомпилир\w*).{0,25}(?:вирус\w*|троян\w*|малвар\w*|бэкдор\w*|эксплойт\w*|кейлоггер\w*|шифровальщик\w*|руткит\w*|рансомвар\w*)', flags),
                re.compile(r'\b(?:вирус\w*|троян\w*|бэкдор\w*|кейлоггер\w*|шифровальщик\w*|руткит\w*|рансомвар\w*|эксплойт\w*)\b', flags),
                re.compile(r'(?:sql|командн\w*|xss|rce)\s*инъекци\w*', flags),
                re.compile(r'reverse\s*shell|шеллкод|shellcode', flags),
            ],
            "Инструкции по созданию оружия": [
                re.compile(r'оружи[еяй]', flags),
                re.compile(r'пистолет|автомат.{0,5}калашников|винтовк|пулемёт|дробовик', flags),
                re.compile(r'(?:созда|сделать|собрать|изготов\w*|напечат\w*).{0,25}(?:оружи[еяй]|пистолет\w*|автомат\w*|ствол\w*)', flags),
                re.compile(r'(?:3d|3д).{0,10}(?:принтер\w*).{0,25}(?:оружи[еяй]|пистолет\w*|ствол\w*)', flags),
                re.compile(r'(?:оружи[еяй]|пистолет\w*).{0,25}(?:3d|3д).{0,10}(?:принтер\w*)', flags),
                re.compile(r'бомб[ауеы]|взрыв|взорв|разбомб|детонатор|взрывчатк', flags),
                re.compile(r'вооружённ|вооружен', flags),
                re.compile(r'\bсву\b|коктейл\w*\s+молотов\w*', flags),
            ],
            "Схемы обмана людей": [
                re.compile(r'(?:как|способы?).{0,20}(?:обмануть|развести|кинуть|облапош\w*).{0,25}(?:людей|человек\w*|клиент\w*|пользовател\w*|деньги|банк\w*)', flags),
                re.compile(r'фишинг\w*|кардинг\w*|дроппер\w*|финансов\w*\s+пирамид\w*', flags),
                re.compile(r'(?:украсть|похит\w*).{0,20}(?:деньги|данные\s+карт\w*|парол\w*)', flags),
                re.compile(r'ограб', flags),
            ],
            "Причинение вреда человеку": [
                re.compile(r'убийств|убить|убью|убей|убьём', flags),
                re.compile(r'казн(?:ить|и[тьюя]|ю|ят|я)', flags),
                re.compile(r'насильственн|насили[еяю]|насиловат', flags),
                re.compile(r'нападени|напасть|напад[уае]', flags),
                re.compile(r'суицид|самоубийств', flags),
                re.compile(r'хедшот', flags),
                re.compile(r'стереть.{0,20}человечеств', flags),
                re.compile(r'отрав', flags),
                re.compile(r'причин\w*.{0,20}вред\w*.{0,20}человек\w*|пытк\w*|пытать|изувеч\w*', flags),
            ],
            "Киберпреступления и харассмент": [
                re.compile(r'(?:взлом\w*|хакн\w*).{0,20}(?:сайт\w*|сервер\w*|баз\w*\s+данных|аккаунт\w*|почт\w*|телефон\w*|систем\w*)', flags),
                re.compile(r'доксинг\w*|доксить|деанон\w*|сталкер\w*|сталкинг\w*', flags),
                re.compile(r'кибербуллинг\w*|харассмент\w*|травл[яею]\s+(?:в\s+сети|онлайн)', flags),
                re.compile(r'(?:слить|опубликова\w*).{0,20}(?:интимн\w*|приватн\w*|персональн\w*).{0,20}(?:фот\w*|данн\w*|видео)', flags),
                re.compile(r'шантаж\w*|вымогат\w*|ddos|ддос', flags),
                re.compile(r'несанкционированн\w*\s+доступ', flags),
            ],
            "Дискриминация": [
                re.compile(
                    r'(?:гее?[вйя]м?|гомосексуал\w*|лесбиянк\w*|транс.людям?|'
                    r'евре[йиеяю]\w*|евреям|мусульман\w*|'
                    r'женщин\w*|инвалид\w*|мигрант\w*|беженц\w*)'
                    r'.{0,40}'
                    r'(?:нельзя|не\s+должн\w*|не\s+позволя\w*|не\s+допуск\w*|запрет\w*)',
                    flags
                ),
                re.compile(
                    r'(?:нельзя|не\s+должн\w*)'
                    r'.{0,30}'
                    r'(?:вступать|находиться|допуск\w*|работать|учиться|голосовать|баллотир\w*)',
                    flags
                ),
                re.compile(
                    r'(?:женщин\w*|мужчин\w*|мигрант\w*|инвалид\w*)'
                    r'.{0,30}'
                    r'зарабатыва\w*.{0,20}меньше',
                    flags
                ),
            ],
        }

    def check_request(self, text: str) -> FilterResult:
        matched_rules = []
        matched_patterns = []

        for category, patterns in self.rules.items():
            for p in patterns:
                if p.search(text):
                    matched_rules.append(category)
                    matched_patterns.append(p.pattern)

        is_blocked = 1 if matched_rules else 0
        return FilterResult(
            request=text,
            is_blocked=is_blocked,
            matched_rules=list(set(matched_rules)),
            matched_patterns=matched_patterns
        )

    def filter_dataframe(self, df: pd.DataFrame, column: str = 'request') -> pd.DataFrame:
        results = df[column].apply(self.check_request)
        df_out = df.copy()
        df_out['prediction'] = results.apply(lambda r: r.is_blocked)
        df_out['matched_rules'] = results.apply(lambda r: r.matched_rules)
        df_out['matched_patterns'] = results.apply(lambda r: r.matched_patterns)
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


    filter_model = RegexFilter()

    start_time = time.perf_counter()
    df_result = filter_model.filter_dataframe(df, column='request')
    end_time = time.perf_counter()


    total_time_ms = (end_time - start_time) * 1000
    latency_ms = total_time_ms / len(df)
    throughput = len(df) / (end_time - start_time)

    metrics = calculate_metrics(df_result['label'], df_result['prediction'])

    print("=== Результаты работы RegEx-фильтра ===")
    print(f"Запросов проверено: {len(df)}")
    print(f"Матрица ошибок: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"Защищенность (TPR / Recall): {metrics['TPR']*100:.1f}%")
    print(f"Ложные срабатывания (FPR):    {metrics['FPR']*100:.1f}%")
    print(f"Точность (Precision):         {metrics['Precision']*100:.1f}%")
    print(f"F1-мера:                      {metrics['F1']:.4f}")
    print(f"Успешность атак (ASR):        {metrics['ASR']*100:.1f}%")
    print(f"Средняя задержка (Latency):   {latency_ms:.4f} мс/запрос")
    print(f"Пропускная способность:       {throughput:.1f} запросов/сек")

    export_df = df_result[['request', 'category', 'label', 'prediction', 'matched_rules']].copy()
    export_df['matched_rules'] = export_df['matched_rules'].apply(lambda x: '; '.join(x) if x else '')
    export_df.to_csv("regex_filter_results.csv", index=False, encoding='utf-8-sig')
    print("\nРезультаты сохранены в regex_filter_results.csv")


if __name__ == '__main__':
    main()
