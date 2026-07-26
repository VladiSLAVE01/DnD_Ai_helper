# generation.py
import pandas as pd
from GAN import DnDGAN
from load_data import load_and_prepare_data
from dnd_rules import calculate_stats_by_rules, compare_with_generated, print_rules_stats

def test_generation():
	data = load_and_prepare_data('dnd_heroes.csv')

	gan = DnDGAN(
		condition_dim=data['condition_dim'],
		feature_dim=data['feature_dim'],
		device='cpu'
	)
	gan.train(data['full_tensor'], epochs=10, batch_size=16)

	# 3. Тестовые герои
	test_cases = [
		('Человек', 'Варвар'),
		('Дварф (Холмовой)', 'Друид'),
		('Человек', 'Маг'),
	]

	results = []
	for race, class_name in test_cases:

		cond = data['create_one_hot_condition'](race, class_name, data)
		hero = gan.generate_hero(cond, data['scaler'], data['target_columns'])
		hero_int = {k: int(round(v)) for k, v in hero.items()}

		print(f"\n Сгенерировано нейросетью:")
		for key, value in hero_int.items():
			print(f"  {key}: {value}")

		rules_stats = calculate_stats_by_rules(race, class_name)

		print(f"\n Рассчитано по правилам D&D:")
		for key, value in rules_stats.items():
			print(f"  {key}: {value}")

		comparison = compare_with_generated(hero_int, rules_stats, race, class_name)

		hero_int['Раса'] = race
		hero_int['Класс'] = class_name
		hero_int['Соответствие'] = comparison['match_level']

		results.append(hero_int)

	df = pd.DataFrame(results)
	df.to_csv('generated_heroes.csv', sep=';', index=False, encoding='utf-8-sig')

	print("\n Сохранено в generated_heroes.csv")
	print(df.to_string(index=False))


if __name__ == "__main__":
	test_generation()