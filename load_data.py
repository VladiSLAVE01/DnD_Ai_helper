# load_data.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path):
	"""
	Загружает CSV в горизонтальном формате с 50+ колонками
	"""
	# ========== 1. Загрузка CSV ==========
	print(f" Загружаем файл: {csv_path}")
	df = pd.read_csv(csv_path, delimiter=';')  # ← точка с запятой!
	print(f"Загружено героев: {len(df)}")
	print(f"Колонок: {len(df.columns)}")

	# ========== 2. Определяем категориальные условия ==========
	condition_columns = ['Раса', 'Класс']
	# Оружие берем из колонок Оружие1 и Оружие2, но пока используем как признак

	# ========== 3. Определяем числовые признаки (для генерации) ==========
	# Основные статы
	target_columns = [
		'Сила', 'Ловкость', 'Телосложение',
		'Интеллект', 'Мудрость', 'Харизма',
		'КД', 'Макс_хиты'
	]

	# ========== 4. Проверка наличия колонок ==========
	all_required = condition_columns + target_columns
	missing_cols = [col for col in all_required if col not in df.columns]

	if missing_cols:
		print(f" Отсутствуют колонки: {missing_cols}")
		print(f"   Доступные колонки: {df.columns.tolist()[:20]}...")
		# Пропускаем недостающие колонки
		for col in missing_cols:
			if col in condition_columns:
				condition_columns.remove(col)
			if col in target_columns:
				target_columns.remove(col)

	print(f"\n Условия: {condition_columns}")
	print(f" Признаки для генерации: {target_columns}")

	# ========== 5. One-Hot кодирование условий ==========
	df_conditions = df[condition_columns].copy()

	# Заменяем NaN на 'Unknown' для категорий
	for col in condition_columns:
		df_conditions[col] = df_conditions[col].fillna('Unknown')

	df_encoded = pd.get_dummies(df_conditions, columns=condition_columns)
	print(f"\n One-Hot условия: {df_encoded.shape[1]} колонок")

	# ========== 6. Нормализация числовых признаков ==========
	df_numeric = df[target_columns].copy()

	# Заменяем NaN на 0 для числовых
	df_numeric = df_numeric.fillna(0)

	# Проверяем, что все колонки числовые
	for col in target_columns:
		df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

	scaler = StandardScaler()
	df_numeric_normalized = pd.DataFrame(
		scaler.fit_transform(df_numeric),
		columns=target_columns
	)

	print(f"\nНормализованные признаки (среднее=0, std=1):")
	for col in target_columns:
		idx = target_columns.index(col)
		print(f"   {col}: mean={scaler.mean_[idx]:.2f}, std={scaler.scale_[idx]:.2f}")

	# ========== 7. Объединение ==========
	full_data = pd.concat([df_encoded, df_numeric_normalized], axis=1)
	full_tensor = torch.tensor(full_data.values.astype(np.float32), dtype=torch.float32)

	# ========== 8. Создаем функцию для One-Hot условия ==========
	def create_one_hot_condition(race, class_name, metadata):
		"""Создает One-Hot вектор для генерации героя"""
		categorical_columns = metadata['categorical_encoder'].columns.tolist()
		condition = np.zeros(len(categorical_columns), dtype=np.float32)

		race_col = f"Раса_{race}"
		class_col = f"Класс_{class_name}"

		if race_col in categorical_columns:
			condition[categorical_columns.index(race_col)] = 1.0
		else:
			print(f"  Раса '{race}' не найдена. Доступны: {[c for c in categorical_columns if 'Раса' in c]}")

		if class_col in categorical_columns:
			condition[categorical_columns.index(class_col)] = 1.0
		else:
			print(f" Класс '{class_name}' не найден. Доступны: {[c for c in categorical_columns if 'Класс' in c]}")

		return condition

	# ========== 9. Метаданные ==========
	metadata = {
		'full_tensor': full_tensor,
		'condition_dim': df_encoded.shape[1],
		'feature_dim': df_numeric_normalized.shape[1],
		'target_columns': target_columns,
		'condition_columns': condition_columns,
		'scaler': scaler,
		'categorical_encoder': df_encoded,
		'df_original': df,
		'num_heroes': len(df),
		'create_one_hot_condition': create_one_hot_condition
	}

	print(f"\n Итоговый тензор: {full_tensor.shape}")
	print(f"   - Условий (One-Hot): {df_encoded.shape[1]}")
	print(f"   - Признаков (числовые): {df_numeric_normalized.shape[1]}")
	print("=" * 50)

	if len(df) < 10:
		print("  ВНИМАНИЕ: Меньше 10 героев! Добавьте больше для качественного обучения.")

	print(" Данные готовы к обучению!")

	return metadata


# =============================================
# ТЕСТ
# =============================================
if __name__ == "__main__":
	try:
		data = load_and_prepare_data('dnd_heroes.csv')

		print(f"\n Данные:")
		print(f"   Всего героев: {data['num_heroes']}")
		print(f"   Условия: {data['condition_columns']}")
		print(f"   Признаки: {data['target_columns']}")

		# Проверка первого героя
		print(f"\n Пример героя (из датасета):")
		first_hero = data['df_original'].iloc[0]
		print(f"   Имя: {first_hero.get('Имя_персонажа', 'Не указано')}")
		print(f"   Раса: {first_hero.get('Раса', 'Не указана')}")
		print(f"   Класс: {first_hero.get('Класс', 'Не указан')}")
		print(f"   Сила: {first_hero.get('Сила', 'Нет')}")
		print(f"   Ловкость: {first_hero.get('Ловкость', 'Нет')}")

		# Тест One-Hot
		print(f"\n Тест One-Hot условия:")
		condition = data['create_one_hot_condition']('Человек', 'Варвар', data)
		print(f"   Размер условия: {len(condition)}")
		print(f"   Активированные колонки:")
		for i, val in enumerate(condition):
			if val == 1.0:
				print(f"      - {data['categorical_encoder'].columns[i]}")

	except Exception as e:
		print(f"\n Ошибка: {e}")
		import traceback

		traceback.print_exc()