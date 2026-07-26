# VAE.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================
# 1. АРХИТЕКТУРА VAE
# =============================================
class Encoder(nn.Module):
	"""Кодировщик: сжимает данные в латентное пространство"""

	def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
		super(Encoder, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.BatchNorm1d(hidden_dim // 2),
			nn.ReLU(True),
			nn.Dropout(0.2),
		)
		# Выходы: среднее и логарифм дисперсии
		self.mu = nn.Linear(hidden_dim // 2, latent_dim)
		self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

	def forward(self, x):
		h = self.model(x)
		mu = self.mu(h)
		logvar = self.logvar(h)
		return mu, logvar


class Decoder(nn.Module):
	"""Декодировщик: восстанавливает данные из латентного пространства"""

	def __init__(self, latent_dim, output_dim, hidden_dim=128):
		super(Decoder, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim // 2),
			nn.BatchNorm1d(hidden_dim // 2),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(hidden_dim // 2, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(True),
			nn.Dropout(0.2),

			nn.Linear(hidden_dim, output_dim),
		)

	def forward(self, z):
		return self.model(z)


class DnDVAE(nn.Module):
	"""
	Вариационный Автоэнкодер для генерации героев D&D
	"""

	def __init__(self, input_dim, latent_dim=16, hidden_dim=128):
		super(DnDVAE, self).__init__()
		self.latent_dim = latent_dim
		self.input_dim = input_dim

		self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
		self.decoder = Decoder(latent_dim, input_dim, hidden_dim)

	def reparameterize(self, mu, logvar):
		"""
		Перепараметризация: z = mu + sigma * eps
		Позволяет обучаться через градиенты
		"""
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		x_hat = self.decoder(z)
		return x_hat, mu, logvar, z

	def generate(self, num_samples, device='cpu', conditions=None):
		"""
		Генерация новых героев из латентного пространства

		Args:
			num_samples: количество героев
			device: 'cuda' или 'cpu'
			conditions: условия (раса, класс) в One-Hot формате (необязательно)

		Returns:
			torch.Tensor: сгенерированные герои
		"""
		self.eval()
		with torch.no_grad():
			# Генерируем случайные латентные векторы
			z = torch.randn(num_samples, self.latent_dim).to(device)

			# Если есть условия — комбинируем с z
			if conditions is not None:
				z = torch.cat([z, conditions], dim=1)

			generated = self.decoder(z)
		return generated


# =============================================
# 2. КЛАСС ДЛЯ ОБУЧЕНИЯ VAE
# =============================================
class DnDVAETrainer:
	def __init__(self, input_dim, latent_dim=16, hidden_dim=128, device='cpu'):
		"""
		Инициализация VAE

		Args:
			input_dim: размерность входных данных (условия + признаки)
			latent_dim: размерность латентного пространства
			hidden_dim: размерность скрытых слоев
			device: 'cuda' или 'cpu'
		"""
		self.device = device
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim

		self.model = DnDVAE(input_dim, latent_dim, hidden_dim).to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		self.history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

		print(f" VAE инициализирован на устройстве: {device}")
		print(f"   - Входная размерность: {input_dim}")
		print(f"   - Латентное пространство: {latent_dim}")
		print(f"   - Скрытых нейронов: {hidden_dim}")

	def vae_loss(self, x, x_hat, mu, logvar, beta=1.0):
		"""
		Функция потерь VAE

		Args:
			x: исходные данные
			x_hat: восстановленные данные
			mu: среднее от энкодера
			logvar: логарифм дисперсии от энкодера
			beta: вес KL-дивергенции (β-VAE)

		Returns:
			recon_loss, kl_loss, total_loss
		"""
		# Реконструкционная ошибка (MSE)
		recon_loss = nn.MSELoss()(x_hat, x)

		# KL-дивергенция (штраф за отклонение от нормального распределения)
		kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

		total_loss = recon_loss + beta * kl_loss
		return total_loss, recon_loss, kl_loss

	def train_epoch(self, dataloader, beta=1.0):
		"""
		Обучает модель на одной эпохе
		"""
		epoch_loss = 0
		epoch_recon_loss = 0
		epoch_kl_loss = 0
		num_batches = 0

		for batch_idx, (data,) in enumerate(dataloader):
			data = data.to(self.device)
			batch_size = data.size(0)

			self.optimizer.zero_grad()

			# Прямой проход
			x_hat, mu, logvar, _ = self.model(data)

			# Расчет потерь
			loss, recon_loss, kl_loss = self.vae_loss(data, x_hat, mu, logvar, beta)

			# Обратный проход
			loss.backward()
			self.optimizer.step()

			epoch_loss += loss.item()
			epoch_recon_loss += recon_loss.item()
			epoch_kl_loss += kl_loss.item()
			num_batches += 1

		return (epoch_loss / num_batches,
		        epoch_recon_loss / num_batches,
		        epoch_kl_loss / num_batches)

	def train(self, full_tensor, epochs=500, batch_size=32, beta=0.1, verbose=True):
		"""
		Полный цикл обучения

		Args:
			full_tensor: тензор с данными
			epochs: количество эпох
			batch_size: размер батча
			beta: вес KL-дивергенции (рекомендуется 0.1-1.0)
			verbose: выводить ли прогресс
		"""
		dataset = TensorDataset(full_tensor)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		print(f"\n Начинаем обучение VAE на {len(full_tensor)} примерах")
		print(f"   Эпох: {epochs}, Батч-сайз: {batch_size}, β: {beta}")


		for epoch in range(1, epochs + 1):
			loss, recon_loss, kl_loss = self.train_epoch(dataloader, beta)

			self.history['loss'].append(loss)
			self.history['recon_loss'].append(recon_loss)
			self.history['kl_loss'].append(kl_loss)

			if verbose and (epoch % 50 == 0 or epoch == 1):
				print(f"Эпоха {epoch:4d}/{epochs} | Loss: {loss:.4f} | Recon: {recon_loss:.4f} | KL: {kl_loss:.4f}")


		print("Обучение VAE завершено")

	def generate_hero(self, num_samples=1, scaler=None, target_columns=None,
	                  condition_dim=None, device='cpu'):
		"""
		Генерирует нового героя из латентного пространства

		Args:
			num_samples: количество героев
			scaler: StandardScaler для денормализации
			target_columns: названия колонок для вывода
			condition_dim: размерность условий (чтобы отделить статы)
			device: 'cuda' или 'cpu'

		Returns:
			dict или list: сгенерированные герои
		"""
		self.model.eval()
		with torch.no_grad():
			z = torch.randn(num_samples, self.latent_dim).to(device)
			generated = self.model.decoder(z)
			generated = generated.cpu().numpy()

		#  Если есть condition_dim — отделяем условия от статов
		if condition_dim is not None and condition_dim > 0:
			# Отделяем условия (первые condition_dim колонок)
			conditions = generated[:, :condition_dim]
			# Берем только статы (остальные колонки)
			generated_stats = generated[:, condition_dim:]
		else:
			generated_stats = generated

		#  Денормализуем только статы
		if scaler is not None:
			generated_stats = scaler.inverse_transform(generated_stats)

		#  Собираем результат
		if target_columns is not None:
			results = []
			for i in range(num_samples):
				hero_dict = dict(zip(target_columns, generated_stats[i]))
				results.append(hero_dict)

			if num_samples == 1:
				return results[0]
			return results
		else:
			return generated_stats

	def plot_history(self):
		"""Визуализирует историю обучения"""
		plt.figure(figsize=(12, 4))

		plt.subplot(1, 3, 1)
		plt.plot(self.history['loss'])
		plt.xlabel('Эпоха')
		plt.ylabel('Total Loss')
		plt.title('Общая ошибка')
		plt.grid(True, alpha=0.3)

		plt.subplot(1, 3, 2)
		plt.plot(self.history['recon_loss'], color='green')
		plt.xlabel('Эпоха')
		plt.ylabel('Reconstruction Loss')
		plt.title('Ошибка реконструкции')
		plt.grid(True, alpha=0.3)

		plt.subplot(1, 3, 3)
		plt.plot(self.history['kl_loss'], color='orange')
		plt.xlabel('Эпоха')
		plt.ylabel('KL Loss')
		plt.title('KL-дивергенция')
		plt.grid(True, alpha=0.3)

		plt.tight_layout()
		plt.show()

	def save(self, path='dnd_vae.pth'):
		"""Сохраняет модель"""
		torch.save({
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'history': self.history,
			'latent_dim': self.latent_dim,
			'input_dim': self.input_dim,
			'hidden_dim': self.hidden_dim,
		}, path)
		print(f"Модель сохранена как {path}")

	def load(self, path='dnd_vae.pth'):
		"""Загружает модель"""
		checkpoint = torch.load(path, map_location=self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.history = checkpoint['history']
		self.latent_dim = checkpoint['latent_dim']
		self.input_dim = checkpoint['input_dim']
		self.hidden_dim = checkpoint['hidden_dim']
		print(f" Модель загружена из {path}")


# =============================================
# 3. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================
if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Загрузка данных
	from load_data import load_and_prepare_data

	data = load_and_prepare_data('dnd_heroes.csv')

	full_tensor = data['full_tensor']
	input_dim = full_tensor.shape[1]  # (условия + признаки)

	print(f"\n Размерность входных данных: {input_dim}")

	# Создаем и обучаем VAE
	vae = DnDVAETrainer(
		input_dim=input_dim,
		latent_dim=8,  # Маленькое латентное пространство для 70 героев
		hidden_dim=64,
		device=device
	)

	# Обучение (500 эпох, небольшой батч)
	vae.train(full_tensor, epochs=500, batch_size=8, beta=0.1)

	# Генерация героев
	# print("\nГенерируем 3 новых героев...")
	# target_cols = ['Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар', 'КД', 'МаксХП']
	# heroes = vae.generate_hero(
	# 	num_samples=3,
	# 	scaler=data['scaler'],
	# 	target_columns=target_cols,
	# 	device=device
	# )
	#
	# for i, hero in enumerate(heroes, 1):
	# 	print(f"\n ГЕРОЙ #{i}")
	# 	for key, value in hero.items():
	# 		print(f"  {key}: {int(round(value))}")

	# График обучения
	vae.plot_history()