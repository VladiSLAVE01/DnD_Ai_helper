# hybrid_generator.py
"""
Гибридный генератор: VAE + правила D&D
"""
import numpy as np
import pandas as pd
from VAE import DnDVAETrainer  # ← Импортируем VAE вместо GAN
from load_data import load_and_prepare_data
from dnd_rules import RACIAL_BONUSES, calculate_modifier, CLASS_BASE


class HybridHeroGenerator:
	"""
	Генерирует героев, используя VAE для креативности
	и правила D&D для валидации
	"""

	def __init__(self, csv_path='dnd_heroes.csv', use_vae=True):
		"""
		Инициализация гибридного генератора

		Args:
			csv_path: путь к CSV с героями
			use_vae: True — использует VAE, False — использует GAN
		"""
		self.use_vae = use_vae
		self.data = load_and_prepare_data(csv_path)
		self.full_tensor = self.data['full_tensor']

		if use_vae:
			# Используем VAE
			print("\n🧠 Инициализация VAE...")
			self.vae = DnDVAETrainer(
				input_dim=self.full_tensor.shape[1],
				latent_dim=8,  # Маленькое латентное пространство
				hidden_dim=64,  # Умеренная сложность
				device='cpu'
			)
			# Обучаем VAE (500 эпох для 70 героев)
			self.vae.train(
				self.full_tensor,
				epochs=500,
				batch_size=8,
				beta=0.1,
				verbose=True
			)
			print("✅ VAE обучен!")
		else:
			# Используем GAN (старый вариант)
			print("\n🧠 Инициализация GAN...")
			from GAN import DnDGAN
			self.gan = DnDGAN(
				condition_dim=self.data['condition_dim'],
				feature_dim=self.data['feature_dim'],
				device='cpu'
			)
			self.gan.train(self.full_tensor, epochs=100, batch_size=16)
			print("✅ GAN обучен!")

	def generate_hero(self, race, class_name):
		"""
		Генерирует валидного героя

		Args:
			race: str (например, 'Человек', 'Дварф (Холмовой)')
			class_name: str (например, 'Варвар', 'Маг')

		Returns:
			dict: валидный герой со статами
		"""
		# 1. Нейросеть генерирует "сырые" статы
		if self.use_vae:
			raw_stats = self._generate_with_vae(race, class_name)
		else:
			raw_stats = self._generate_with_gan(race, class_name)

		print(f"\n🧠 Нейросеть предложила:")
		for key, value in raw_stats.items():
			print(f"  {key}: {value}")

		# 2. Применяем правила D&D
		validated_stats = self._apply_rules(raw_stats, race, class_name)

		print(f"\n📖 После применения правил D&D:")
		for key, value in validated_stats.items():
			print(f"  {key}: {value}")

		return validated_stats

	def _generate_with_vae(self, race, class_name):
		"""
		Генерация через VAE с учетом условий
		"""
		# Получаем размерность условий из данных
		condition_dim = self.data.get('condition_dim', 0)

		# Генерируем 5 кандидатов
		candidates = self.vae.generate_hero(
			num_samples=5,
			scaler=self.data['scaler'],
			target_columns=self.data['target_columns'],
			condition_dim=condition_dim,  # ← Передаем размерность условий
			device='cpu'
		)

		# Если candidates — это список словарей
		if isinstance(candidates, list):
			# Выбираем героя с наиболее подходящими статами для класса
			best_hero = self._select_best_for_class(candidates, class_name)
		else:
			best_hero = candidates

		# Превращаем в словарь с короткими ключами
		short_keys = {'Сила': 'Сил', 'Ловкость': 'Лов', 'Телосложение': 'Тел',
		              'Интеллект': 'Инт', 'Мудрость': 'Муд', 'Харизма': 'Хар',
		              'КД': 'КД', 'МаксХП': 'МаксХП'}

		raw_stats = {}
		for full_key, value in best_hero.items():
			if full_key in short_keys:
				raw_stats[short_keys[full_key]] = int(round(value))
			else:
				raw_stats[full_key] = int(round(value))

		return raw_stats

	def _generate_with_gan(self, race, class_name):
		"""
		Генерация через GAN (старый метод)
		"""
		cond = self.data['create_one_hot_condition'](race, class_name, self.data)
		raw_hero = self.gan.generate_hero(cond, self.data['scaler'], self.data['target_columns'])
		raw_stats = {k: int(round(v)) for k, v in raw_hero.items()}
		return raw_stats

	def _select_best_for_class(self, candidates, class_name):
		"""
		Выбирает кандидата с наиболее подходящими статами для класса
		"""
		# Основная характеристика для класса
		primary_stat = self._get_primary_stat(class_name)

		if primary_stat is None:
			return candidates[0]

		# Оцениваем каждого кандидата
		best_score = -float('inf')
		best_candidate = candidates[0]

		for hero in candidates:
			score = hero.get(primary_stat, 0)
			# Добавляем бонус за сбалансированность
			stats = [hero.get('Сила', 0), hero.get('Ловкость', 0),
			         hero.get('Телосложение', 0), hero.get('Интеллект', 0),
			         hero.get('Мудрость', 0), hero.get('Харизма', 0)]
			# Штраф за слишком низкие статы
			if any(s < 6 for s in stats):
				score -= 10
			# Бонус за разнообразие
			score += np.std(stats) * 0.5

			if score > best_score:
				best_score = score
				best_candidate = hero

		return best_candidate

	def _apply_rules(self, raw_stats, race, class_name):
		"""
		Применяет правила D&D к сгенерированным статам

		1. Проверяет диапазон (3-20)
		2. Добавляет бонусы расы
		3. Проверяет требования класса
		4. Корректирует, если нужно
		"""
		# Маппинг ключей
		KEY_MAP = {'Сил': 'Сила', 'Лов': 'Ловкость', 'Тел': 'Телосложение',
		           'Инт': 'Интеллект', 'Муд': 'Мудрость', 'Хар': 'Харизма'}
		REVERSE_MAP = {v: k for k, v in KEY_MAP.items()}

		# 2.1 Создаем словарь с полными именами
		stats = {}
		for short_key, value in raw_stats.items():
			if short_key in KEY_MAP:
				stats[KEY_MAP[short_key]] = value

		# 2.2 Применяем бонусы расы
		if race in RACIAL_BONUSES:
			for stat, bonus in RACIAL_BONUSES[race].items():
				if stat in stats:
					stats[stat] += bonus

		# 2.3 Проверяем диапазон 3-20
		for stat in stats:
			stats[stat] = max(3, min(20, stats[stat]))

		# 2.4 Проверяем требования класса
		if class_name in CLASS_BASE:
			required_stat = self._get_primary_stat(class_name)
			min_value = 13

			if required_stat and stats.get(required_stat, 0) < min_value:
				print(f"  ⚠️ {required_stat} слишком низкий для {class_name} ({stats[required_stat]} < {min_value})")
				stats[required_stat] = min_value
				self._rebalance_stats(stats, required_stat)

		# 2.5 Переводим обратно в короткие ключи для совместимости
		result = {}
		for full_key, value in stats.items():
			if full_key in REVERSE_MAP:
				result[REVERSE_MAP[full_key]] = value
			else:
				result[full_key] = value

		# 2.6 Добавляем производные параметры
		dex_mod = calculate_modifier(result.get('Лов', 10))
		con_mod = calculate_modifier(result.get('Тел', 10))
		result['КД'] = 10 + dex_mod
		result['МаксХП'] = self._calculate_hp(class_name, result.get('Тел', 10))

		return result

	def _get_primary_stat(self, class_name):
		"""Возвращает основную характеристику для класса"""
		primary_stats = {
			'Варвар': 'Сила',
			'Друид': 'Мудрость',
			'Маг': 'Интеллект',
			'Жрец': 'Мудрость',
			'Разбойник': 'Ловкость',
			'Паладин': 'Сила',
			'Чародей': 'Харизма',
			'Колдун': 'Харизма',
			'Монах': 'Мудрость',
			'Следопыт': 'Ловкость',
			'Бард': 'Харизма',
			'Воин': 'Сила',
			'Волшебник': 'Интеллект',  # Добавляем для совместимости
		}
		return primary_stats.get(class_name, None)

	def _calculate_hp(self, class_name, constitution):
		"""Расчет ХП для 1 уровня"""
		hit_dice = {
			'Варвар': 12, 'Воин': 10, 'Паладин': 10, 'Следопыт': 10,
			'Друид': 8, 'Жрец': 8, 'Разбойник': 8, 'Монах': 8, 'Бард': 8,
			'Маг': 6, 'Чародей': 6, 'Колдун': 6, 'Волшебник': 6,
		}
		dice = hit_dice.get(class_name, 8)
		con_mod = calculate_modifier(constitution)
		return dice + con_mod

	def _rebalance_stats(self, stats, boosted_stat):
		"""
		Упрощенное перераспределение: забираем 2 очка из самой высокой
		характеристики и отдаем в boosted_stat
		"""
		max_stat = None
		max_value = 0
		for stat, value in stats.items():
			if stat != boosted_stat and value > max_value:
				max_value = value
				max_stat = stat

		if max_stat and max_value > 14:
			stats[max_stat] -= 2
			stats[boosted_stat] += 2


# =============================================
# ТЕСТ
# =============================================
if __name__ == "__main__":
	print("=" * 60)
	print("🎲 ГИБРИДНЫЙ ГЕНЕРАТОР (VAE + ПРАВИЛА D&D)")
	print("=" * 60)

	# Создаем генератор с VAE
	generator = HybridHeroGenerator('dnd_heroes.csv', use_vae=True)

	test_cases = [
		('Человек', 'Варвар'),
		('Дварф (Холмовой)', 'Друид'),
		('Человек', 'Волшебник'),
		('Эльф Высший', 'Волшебник'),
		('Тифлинг', 'Бард'),
	]

	results = []
	for race, class_name in test_cases:
		print("\n" + "=" * 60)
		print(f"🎲 ГЕНЕРАЦИЯ: {race} - {class_name}")
		print("=" * 60)

		hero = generator.generate_hero(race, class_name)
		hero['Раса'] = race
		hero['Класс'] = class_name
		results.append(hero)

	# Сохраняем результаты
	df = pd.DataFrame(results)

	# Переупорядочиваем колонки для красоты
	columns_order = ['Раса', 'Класс', 'Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар', 'КД', 'МаксХП']
	available_cols = [col for col in columns_order if col in df.columns]
	df = df[available_cols]

	df.to_csv('hybrid_heroes_vae.csv', sep=';', index=False, encoding='utf-8-sig')
	print("\n💾 Результаты сохранены в 'hybrid_heroes_vae.csv'")

	print("\n📊 ИТОГОВЫЕ ГЕРОИ:")
	print("=" * 60)
	print(df.to_string(index=False))