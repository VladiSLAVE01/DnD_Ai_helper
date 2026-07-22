import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =============================================
# 1. АРХИТЕКТУРА ГЕНЕРАТОРА
# =============================================
class Generator(nn.Module):
	"""
	Генератор: принимает шум + условия и создает фейковые признаки героя
	"""

	def __init__(self, noise_dim, condition_dim, feature_dim):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			# Вход: шум + условия
			nn.Linear(noise_dim + condition_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(1024, feature_dim),
			# Выход: сгенерированные признаки (сила, ловкость, хп...)
		)

	def forward(self, noise, conditions):
		# Объединяем шум и условия
		x = torch.cat([noise, conditions], dim=1)
		return self.model(x)


# =============================================
# 2. АРХИТЕКТУРА ДИСКРИМИНАТОРА
# =============================================
class Discriminator(nn.Module):
	"""
	Дискриминатор: принимает признаки + условия и решает, реальные ли они
	"""

	def __init__(self, feature_dim, condition_dim):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			# Вход: признаки + условия
			nn.Linear(feature_dim + condition_dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),

			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),

			nn.Linear(256, 128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),

			nn.Linear(128, 1),
			nn.Sigmoid()  # Вероятность, что данные реальные (0-1)
		)

	def forward(self, features, conditions):
		# Объединяем признаки и условия
		x = torch.cat([features, conditions], dim=1)
		return self.model(x)


# =============================================
# 3. КЛАСС ДЛЯ ОБУЧЕНИЯ GAN
# =============================================
class DnDGAN:
	def __init__(self, condition_dim, feature_dim, noise_dim=100, device='cpu'):
		"""
		Инициализация GAN

		Args:
			condition_dim: размерность вектора условий (раса+класс+оружие в One-Hot)
			feature_dim: размерность генерируемых признаков (сила, ловкость...)
			noise_dim: размерность шума (обычно 100)
			device: 'cuda' или 'cpu'
		"""
		self.device = device
		self.noise_dim = noise_dim
		self.condition_dim = condition_dim
		self.feature_dim = feature_dim

		# Создаем генератор и дискриминатор
		self.generator = Generator(noise_dim, condition_dim, feature_dim).to(device)
		self.discriminator = Discriminator(feature_dim, condition_dim).to(device)

		# Оптимизаторы
		self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

		# Функция потерь
		self.criterion = nn.BCELoss()

		# История обучения
		self.history = {'G_loss': [], 'D_loss': []}

		print(f"GAN инициализирован на устройстве: {device}")
		print(f"   - Размер шума: {noise_dim}")
		print(f"   - Размер условия: {condition_dim}")
		print(f"   - Размер признаков: {feature_dim}")

	def train_epoch(self, dataloader):
		"""
		Обучает модель на одной эпохе

		Returns:
			avg_G_loss, avg_D_loss: средние потери за эпоху
		"""
		epoch_G_loss = 0
		epoch_D_loss = 0
		num_batches = 0

		for batch_idx, (real_data,) in enumerate(dataloader):
			batch_size = real_data.size(0)
			real_data = real_data.to(self.device)

			# Разделяем на условия и признаки
			conditions = real_data[:, :self.condition_dim]
			real_features = real_data[:, self.condition_dim:]

			# Создаем метки для BCELoss
			real_labels = torch.ones(batch_size, 1).to(self.device)
			fake_labels = torch.zeros(batch_size, 1).to(self.device)

			# ========== ОБУЧЕНИЕ ДИСКРИМИНАТОРА ==========
			# 1. Обучаем на реальных данных
			self.optimizer_D.zero_grad()
			output_real = self.discriminator(real_features, conditions)
			loss_real = self.criterion(output_real, real_labels)

			# 2. Обучаем
			noise = torch.randn(batch_size, self.noise_dim).to(self.device)
			fake_features = self.generator(noise, conditions)
			output_fake = self.discriminator(fake_features.detach(), conditions)
			loss_fake = self.criterion(output_fake, fake_labels)

			# Общий loss дискриминатора
			loss_D = loss_real + loss_fake
			loss_D.backward()
			self.optimizer_D.step()

			# ========== ОБУЧЕНИЕ ГЕНЕРАТОРА ==========
			self.optimizer_G.zero_grad()
			noise = torch.randn(batch_size, self.noise_dim).to(self.device)
			fake_features = self.generator(noise, conditions)
			output_fake = self.discriminator(fake_features, conditions)

			# Генератор хочет обмануть дискриминатор
			loss_G = self.criterion(output_fake, real_labels)
			loss_G.backward()
			self.optimizer_G.step()

			# Сохраняем потери
			epoch_G_loss += loss_G.item()
			epoch_D_loss += loss_D.item()
			num_batches += 1

		return epoch_G_loss / num_batches, epoch_D_loss / num_batches

	def train(self, full_tensor, epochs=100, batch_size=64, verbose=True):
		"""
		Полный цикл обучения

		Args:
			full_tensor: тензор с данными (условия + признаки)
			epochs: количество эпох
			batch_size: размер батча
			verbose: выводить ли прогресс
		"""
		# Создаем DataLoader
		dataset = TensorDataset(full_tensor)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		print(f"\n Начинаем обучение на {len(full_tensor)} примерах")
		print(f"   Эпох: {epochs}, Батч-сайз: {batch_size}")
		print("-" * 50)

		for epoch in range(1, epochs + 1):
			g_loss, d_loss = self.train_epoch(dataloader)

			# Сохраняем историю
			self.history['G_loss'].append(g_loss)
			self.history['D_loss'].append(d_loss)

			# Логирование
			if verbose and (epoch % 100 == 0 or epoch == 1):
				print(f"Эпоха {epoch:4d}/{epochs} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

			# Ранняя остановка при расходимости
			if g_loss > 10 or d_loss > 10:
				print(f"  Ранняя остановка на эпохе {epoch} (потери слишком большие)")
				break

		print("-" * 50)
		print(" Обучение завершено!")

	def generate_hero(self, condition_vector, scaler=None, target_columns=None):
		"""
		Генерирует нового героя по заданным условиям

		Args:
			condition_vector: numpy array или list с условиями (One-Hot)
			scaler: StandardScaler для денормализации
			target_columns: названия колонок для вывода

		Returns:
			dict: сгенерированные признаки
		"""
		self.generator.eval()
		with torch.no_grad():
			# Преобразуем в тензор
			condition = torch.tensor(condition_vector, dtype=torch.float32).to(self.device)
			condition = condition.unsqueeze(0)  # Добавляем размер батча

			# Генерируем шум
			noise = torch.randn(1, self.noise_dim).to(self.device)

			# Генерируем признаки
			generated_features = self.generator(noise, condition)
			generated_features = generated_features.cpu().numpy().flatten()

		# Денормализуем, если передан scaler
		if scaler is not None:
			generated_features = scaler.inverse_transform(generated_features.reshape(1, -1)).flatten()

		# Создаем словарь с результатом
		if target_columns is not None:
			result = dict(zip(target_columns, generated_features))
		else:
			result = {'generated_features': generated_features}

		return result

	def plot_history(self):
		"""Визуализирует историю обучения"""
		plt.figure(figsize=(12, 5))

		plt.subplot(1, 2, 1)
		plt.plot(self.history['G_loss'], label='Generator Loss', color='blue')
		plt.plot(self.history['D_loss'], label='Discriminator Loss', color='red')
		plt.xlabel('Эпоха')
		plt.ylabel('Потери (Loss)')
		plt.title('История обучения GAN')
		plt.legend()
		plt.grid(True, alpha=0.3)

		plt.subplot(1, 2, 2)
		plt.plot(self.history['G_loss'], label='G Loss', color='blue')
		plt.plot(self.history['D_loss'], label='D Loss', color='red')
		plt.xlabel('Эпоха')
		plt.ylabel('Потери (Loss)')
		plt.title('Увеличенный график')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.ylim(0, 5)

		plt.tight_layout()
		plt.show()

	def save(self, path_prefix='dnd_gan'):
		"""Сохраняет модель"""
		torch.save({
			'generator_state_dict': self.generator.state_dict(),
			'discriminator_state_dict': self.discriminator.state_dict(),
			'optimizer_G_state_dict': self.optimizer_G.state_dict(),
			'optimizer_D_state_dict': self.optimizer_D.state_dict(),
			'history': self.history,
			'noise_dim': self.noise_dim,
			'condition_dim': self.condition_dim,
			'feature_dim': self.feature_dim,
		}, f'{path_prefix}_checkpoint.pth')
		print(f" Модель сохранена как {path_prefix}_checkpoint.pth")

	def load(self, path='dnd_gan_checkpoint.pth'):
		"""Загружает модель"""
		checkpoint = torch.load(path, map_location=self.device)
		self.generator.load_state_dict(checkpoint['generator_state_dict'])
		self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
		self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
		self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
		self.history = checkpoint['history']
		print(f" Модель загружена из {path}")


def create_one_hot_condition(race, class_name, metadata):
	"""Создает One-Hot вектор для генерации героя"""
	categorical_columns = metadata['categorical_encoder'].columns.tolist()
	condition = np.zeros(len(categorical_columns), dtype=np.float32)

	race_col = f"Раса_{race}"
	class_col = f"Класс_{class_name}"

	if race_col in categorical_columns:
		condition[categorical_columns.index(race_col)] = 1.0
	else:
		print(f"  Раса '{race}' не найдена")

	if class_col in categorical_columns:
		condition[categorical_columns.index(class_col)] = 1.0
	else:
		print(f"️  Класс '{class_name}' не найден")

	return condition


# В блоке if __name__ == "__main__": измените:
if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Загрузка данных
	from load_data import load_and_prepare_data

	data = load_and_prepare_data('dnd_heroes.csv')

	full_data = data['full_tensor']
	condition_dim = data['condition_dim']
	feature_dim = data['feature_dim']
	scaler = data['scaler']
	target_columns = data['target_columns']

	print(f"\nЗагружено {len(full_data)} героев")

	# Создаем GAN
	gan = DnDGAN(
		condition_dim=condition_dim,
		feature_dim=feature_dim,
		noise_dim=100,
		device=device
	)

	# ОБУЧЕНИЕ
	gan.train(full_data, epochs=10, batch_size=16)

	# ГЕНЕРАЦИЯ
	print("\n Генерируем нового героя...")

	# Создаем условие
	race = "Человек"
	class_name = "Варвар"

	condition = create_one_hot_condition(race, class_name, data)

	hero = gan.generate_hero(
		condition_vector=condition,
		scaler=scaler,
		target_columns=target_columns
	)

	hero_int = {k: int(round(v)) for k, v in hero.items()}

	print(f"\n Сгенерированный герой:")
	print(f"   Раса: {race}")
	print(f"   Класс: {class_name}")
	print("-" * 30)
	for key, value in hero_int.items():
		print(f"   {key}: {value}")

	gan.plot_history()