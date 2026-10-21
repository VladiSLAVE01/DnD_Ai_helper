# load_data.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path):
	print(f"Загружаем: {csv_path}")

	# Загрузка с жесткой проверкой колонок
	try:
		df = pd.read_csv(
			csv_path,
			delimiter=';',
			encoding='utf-8-sig',
			on_bad_lines='skip',  # Пропускаем строки с ошибками
			dtype=str,  # Читаем всё как строки, чтобы не было ошибок с float
			keep_default_na=False  # Пустые строки не превращаем в NaN
		)
	except Exception as e:
		print(f"Ошибка чтения: {e}")
		raise

	# Убираем возможные пробелы в названиях колонок
	df.columns = df.columns.str.strip()

	print(f"Загружено героев: {len(df)}")
	print(f"Колонок: {len(df.columns)}")

	print(f" Загружено героев: {len(df)}")
	print(f"Колонок: {len(df.columns)}")

	# ===== КОЛОНКИ =====
	condition_columns = ['Раса', 'Класс']
	target_columns = ['Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар', 'КД', 'МаксХП']

	# Проверяем наличие колонок
	available = df.columns.tolist()

	# Находим реальные колонки
	actual_conditions = [col for col in condition_columns if col in available]
	actual_targets = [col for col in target_columns if col in available]

	print(f"Условия: {actual_conditions}")
	print(f"Признаки: {actual_targets}")

	# Если колонки не найдены — ищем похожие
	if not actual_conditions:
		for col in condition_columns:
			for av in available:
				if col.lower() in av.lower():
					actual_conditions.append(av)
					break

	if not actual_targets:
		for col in target_columns:
			for av in available:
				if col.lower() in av.lower() or av.lower() in col.lower():
					actual_targets.append(av)
					break

	# ===== ONE-HOT =====
	df_conditions = df[actual_conditions].fillna('Unknown').astype(str)
	df_encoded = pd.get_dummies(df_conditions, columns=actual_conditions)

	#  СБРАСЫВАЕМ ИНДЕКС
	df_encoded = df_encoded.reset_index(drop=True)

	# ===== НОРМАЛИЗАЦИЯ =====
	df_numeric = df[actual_targets].apply(pd.to_numeric, errors='coerce').fillna(0)
	scaler = StandardScaler()
	df_norm = pd.DataFrame(scaler.fit_transform(df_numeric), columns=actual_targets)

	# СБРАСЫВАЕМ ИНДЕКС
	df_norm = df_norm.reset_index(drop=True)

	# ===== ОБЪЕДИНЕНИЕ =====
	full_data = pd.concat([df_encoded, df_norm], axis=1)
	full_tensor = torch.tensor(full_data.values.astype(np.float32))

	# ===== ФУНКЦИЯ УСЛОВИЯ =====
	def create_one_hot_condition(race, class_name, metadata):
		cols = metadata['categorical_encoder'].columns.tolist()
		cond = np.zeros(len(cols), dtype=np.float32)

		# Ищем колонку с таким значением
		for val, prefix in [(race, 'Раса'), (class_name, 'Класс')]:
			target_col = f"{prefix}_{val}"
			if target_col in cols:
				cond[cols.index(target_col)] = 1.0
			return cond

	metadata = {
		'full_tensor': full_tensor,
		'condition_dim': df_encoded.shape[1],
		'feature_dim': df_norm.shape[1],
		'target_columns': actual_targets,
		'scaler': scaler,
		'categorical_encoder': df_encoded,
		'create_one_hot_condition': create_one_hot_condition,
		'num_heroes': len(df),
		'df_original': df
	}

	print(f" Тензор: {full_tensor.shape}")
	print(f" Условий: {df_encoded.shape[1]}, Признаков: {df_norm.shape[1]}")

	# Выводим список доступных рас и классов
	print(f"\n Доступные расы: {df['Раса'].unique().tolist()}")
	print(f" Доступные классы: {df['Класс'].unique().tolist()}")

	return metadata