import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_prepare_data(csv_path, target_columns=None, condition_columns=None):
	"""
	Загружает CSV и подготавливает данные для обучения GAN

	Args:
		csv_path (str): путь к CSV-файлу
		target_columns (list): колонки, которые мы будем генерировать (числовые статы)
		condition_columns (list): колонки-условия (раса, класс, оружие)

	Returns:
		dict: словарь с подготовленными данными
	"""

	# ========== 1. Загрузка CSV ==========
	print(f" Загружаем файл: {csv_path}")
	df = pd.read_csv(csv_path)

	print(f" Загружено строк: {len(df)}")
	print(f" Колонки: {df.columns.tolist()}")
	print("\nПервые 5 строк:")
	print(df.head())

	# ========== 2. Определяем колонки (если не указаны) ==========
	if condition_columns is None:
		# Автоматически определяем текстовые колонки как условия
		condition_columns = df.select_dtypes(include=['object','string']).columns.tolist()

	if target_columns is None:
		# Автоматически определяем числовые колонки как целевые
		target_columns = df.select_dtypes(include=[np.number]).columns.tolist()

	print(f" Условия (входные параметры): {condition_columns}")
	print(f"Целевые параметры (будем генерировать): {target_columns}")

	# ========== 3. Обработка категориальных колонок (One-Hot Encoding) ==========
	categorical_data = df[condition_columns].copy()

	# One-Hot кодирование для категориальных признаков
	categorical_encoded = pd.get_dummies(categorical_data, columns=condition_columns)
	print(f"\n📊 После One-Hot кодирования: {categorical_encoded.shape[1]} признаков")

	# ========== 4. Обработка числовых колонок (нормализация) ==========
	numeric_data = df[target_columns].copy()

	# Сохраняем статистики для денормализации (понадобится при генерации)
	scaler = StandardScaler()
	numeric_normalized = scaler.fit_transform(numeric_data)
	numeric_normalized = pd.DataFrame(numeric_normalized, columns=target_columns)

	print(f"\n Числовые данные нормализованы (mean=0, std=1)")

	# ========== 5. Объединяем всё в один тензор ==========
	# Конкатенируем категориальные (One-Hot) и числовые (нормализованные) данные
	full_data = pd.concat([categorical_encoded, numeric_normalized], axis=1)

	# Преобразуем в numpy, затем в torch tensor
	full_data_np = full_data.values.astype(np.float32)
	full_tensor = torch.tensor(full_data_np, dtype=torch.float32)

	print(f"\n Итоговый тензор: {full_tensor.shape}")
	print(f"   - {categorical_encoded.shape[1]} колонок-условий (One-Hot)")
	print(f"   - {numeric_normalized.shape[1]} колонок-признаков (нормализованные)")
	print(f"   - Всего: {full_tensor.shape[1]} колонок")

	# ========== 6. Разделяем на условие и признаки ==========
	condition_dim = categorical_encoded.shape[1]
	feature_dim = numeric_normalized.shape[1]

	conditions = full_tensor[:, :condition_dim]  # Условия (раса, класс...)
	features = full_tensor[:, condition_dim:]  # Признаки (сила, ловкость...)

	print(f"\n Размерность условия: {condition_dim}")
	print(f" Размерность признаков: {feature_dim}")

	# ========== 7. Сохраняем метаданные ==========
	metadata = {
		'condition_columns': condition_columns,
		'target_columns': target_columns,
		'condition_dim': condition_dim,
		'feature_dim': feature_dim,
		'scaler': scaler,  # Для обратной нормализации
		'categorical_columns': categorical_encoded.columns.tolist(),
		'df_original': df,
		'full_tensor': full_tensor,
		'conditions': conditions,
		'features': features,
		'categorical_encoder': categorical_encoded  # Для декодирования
	}

	return metadata


# ========== ИСПОЛЬЗОВАНИЕ ==========
if __name__ == "__main__":
	# Путь к вашему CSV-файлу
	csv_path = "dnd_heroes.csv"  # Замените на ваш путь

	# Загружаем данные
	data = load_and_prepare_data(csv_path)

	# Проверяем, что получилось
	print("\n" + "=" * 50)
	print(f"Количество героев: {data['full_tensor'].shape[0]}")
	print(f"Размер батча рекомендуется: 32 или 64")
	print(f"Размер шума (noise_dim): 100 (стандарт)")
	print(f"Размер условия (condition_dim): {data['condition_dim']}")
	print(f"Размер признаков (feature_dim): {data['feature_dim']}")