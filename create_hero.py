# create_hero.py
"""
Интерактивный генератор персонажей D&D
Вы вводите расу и класс, ИИ создает героя
"""

import pandas as pd
from hybrid_generator import HybridHeroGenerator
from dnd_rules import RACIAL_BONUSES, CLASS_BASE


def get_available_races(data):
	"""Получает список доступных рас из датасета"""
	if 'df_original' in data:
		return data['df_original']['Раса'].unique().tolist()
	return list(RACIAL_BONUSES.keys())


def get_available_classes(data):
	"""Получает список доступных классов из датасета"""
	if 'df_original' in data:
		return data['df_original']['Класс'].unique().tolist()
	return list(CLASS_BASE.keys())


def print_hero(hero, race, class_name):
	"""Красивый вывод героя"""
	print("\n" + "=" * 60)
	print("🎲 НОВЫЙ ПЕРСОНАЖ СОЗДАН!")
	print("=" * 60)
	print(f"  Раса: {race}")
	print(f"  Класс: {class_name}")
	print("-" * 60)

	# Основные статы
	stats = ['Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар']
	print("  📊 ХАРАКТЕРИСТИКИ:")
	for stat in stats:
		if stat in hero:
			value = hero[stat]
			# Вычисляем модификатор
			mod = (value - 10) // 2
			mod_str = f"{mod:+d}"
			print(f"    {stat}: {value:2d} (модификатор {mod_str})")

	print("-" * 60)

	# Боевые параметры
	combat = ['КД', 'МаксХП']
	print("  ⚔️ БОЕВЫЕ ПАРАМЕТРЫ:")
	for stat in combat:
		if stat in hero:
			print(f"    {stat}: {hero[stat]}")

	# Навыки (если есть)
	skills = ['Акр', 'Атл', 'Восп', 'Выж', 'Выст', 'Зап', 'Ист', 'ЛовР',
	          'Маг', 'Мед', 'Обм', 'Прир', 'Прониц', 'Рассл', 'Религ', 'Скрыт', 'Убеж', 'УходЖ']

	existing_skills = {k: v for k, v in hero.items() if k in skills}
	if existing_skills:
		print("-" * 60)
		print("  🎯 НАВЫКИ:")
		for skill, value in existing_skills.items():
			print(f"    {skill}: {value:+d}")

	print("=" * 60)


def main():
	print("=" * 60)
	print("🧙 ГЕНЕРАТОР ПЕРСОНАЖЕЙ D&D")
	print("=" * 60)
	print("\nЗагрузка модели...")

	# Загружаем генератор
	generator = HybridHeroGenerator('dnd_heroes.csv', use_vae=True)

	# Получаем доступные расы и классы
	races = get_available_races(generator.data)
	classes = get_available_classes(generator.data)

	print("\n✅ Модель загружена!")

	# Основной цикл
	while True:
		print("\n" + "-" * 60)
		print("📋 ДОСТУПНЫЕ РАСЫ:")
		for i, race in enumerate(races, 1):
			print(f"  {i}. {race}")

		print("\n📋 ДОСТУПНЫЕ КЛАССЫ:")
		for i, class_name in enumerate(classes, 1):
			print(f"  {i}. {class_name}")

		print("\n" + "-" * 60)

		# Ввод расы
		while True:
			race_input = input("\n👤 Введите расу (или номер): ").strip()

			# Проверяем, может быть это номер
			if race_input.isdigit():
				idx = int(race_input) - 1
				if 0 <= idx < len(races):
					race = races[idx]
					break
				else:
					print(f"⚠️  Номер должен быть от 1 до {len(races)}")
					continue

			# Проверяем точное совпадение
			if race_input in races:
				race = race_input
				break

			# Ищем частичное совпадение
			matches = [r for r in races if race_input.lower() in r.lower()]
			if len(matches) == 1:
				race = matches[0]
				print(f"✅ Найдено: {race}")
				break
			elif len(matches) > 1:
				print(f"⚠️  Найдено несколько совпадений: {matches}")
				print("   Уточните запрос")
				continue
			else:
				print(f"⚠️  Раса '{race_input}' не найдена")
				print(f"   Доступные: {', '.join(races)}")

		# Ввод класса
		while True:
			class_input = input("⚔️ Введите класс (или номер): ").strip()

			if class_input.isdigit():
				idx = int(class_input) - 1
				if 0 <= idx < len(classes):
					class_name = classes[idx]
					break
				else:
					print(f"⚠️  Номер должен быть от 1 до {len(classes)}")
					continue

			if class_input in classes:
				class_name = class_input
				break

			matches = [c for c in classes if class_input.lower() in c.lower()]
			if len(matches) == 1:
				class_name = matches[0]
				print(f"✅ Найдено: {class_name}")
				break
			elif len(matches) > 1:
				print(f"⚠️  Найдено несколько совпадений: {matches}")
				print("   Уточните запрос")
				continue
			else:
				print(f"⚠️  Класс '{class_input}' не найден")
				print(f"   Доступные: {', '.join(classes)}")

		# Генерируем героя
		print(f"\n🎲 Генерация {race} - {class_name}...")
		hero = generator.generate_hero(race, class_name)

		# Выводим результат
		print_hero(hero, race, class_name)

		# Сохраняем в CSV
		try:
			df = pd.DataFrame([hero])
			df.insert(0, 'Класс', class_name)
			df.insert(0, 'Раса', race)
			df.to_csv('my_heroes.csv', mode='a', header=not pd.io.common.file_exists('my_heroes.csv'),
			          sep=';', index=False, encoding='utf-8-sig')
			print("\n💾 Герой сохранён в 'my_heroes.csv'")
		except:
			pass

		# Спрашиваем, хочет ли пользователь создать ещё
		print("\n" + "-" * 60)
		again = input("Хотите создать ещё одного героя? (да/нет): ").strip().lower()
		if again not in ['да', 'yes', 'y', 'д', 'lf']:
			print("\n✨ Спасибо за использование! Удачи в приключениях! 🎲")
			break


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n\n✨ До свидания! 🎲")
	except Exception as e:
		print(f"\n❌ Ошибка: {e}")
		import traceback

		traceback.print_exc()