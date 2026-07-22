# generation.py
import numpy as np
import pandas as pd
from GAN import DnDGAN
from load_data import load_and_prepare_data


def test_generation():
	# 1. Загружаем данные
	data = load_and_prepare_data('dnd_heroes.csv')
	full_data = data['full_tensor']

	# 2. Создаем GAN
	gan = DnDGAN(
		condition_dim=data['condition_dim'],
		feature_dim=data['feature_dim'],
		device='cpu'
	)
	gan.train(full_data, epochs=10, batch_size=16)

	# 3. Создаем условие
	race = "Человек"
	class_name = "Варвар"

	condition = data['create_one_hot_condition'](race, class_name, data)

	# 4. Генерируем героя
	hero = gan.generate_hero(
		condition_vector=condition,
		scaler=data['scaler'],
		target_columns=data['target_columns']
	)

	# 5. Округляем
	hero_int = {k: int(round(v)) for k, v in hero.items()}

	print("\n" + "=" * 50)
	print(" НОВЫЙ ГЕРОЙ")
	print("=" * 50)
	print(f"Раса: {race}")
	print(f"Класс: {class_name}")
	print("-" * 50)
	for key, value in hero_int.items():
		print(f"{key}: {value}")
	print("=" * 50)

	return hero_int


if __name__ == "__main__":
	test_generation()