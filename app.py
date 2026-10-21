# app.py
"""
FastAPI-сервер для генерации персонажей D&D.
Запуск: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import time
import os


# НАСТРОЙКА ЛОГИРОВАНИЯ

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ИНИЦИАЛИЗАЦИЯ FASTAPI

app = FastAPI(
	title="D&D AI Character Generator API",
	description="Генерация персонажей D&D 5e с помощью VAE + правил игры",
	version="1.0.0"
)

# CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)


# ЗАГРУЗКА ГЕНЕРАТОРА

logger.info(" Загрузка гибридного генератора...")
start_time = time.time()

try:
	from hybrid_generator import HybridHeroGenerator

	#⚠Модель обучается/загружается ОДИН раз при старте сервера
	generator = HybridHeroGenerator('dnd_heroes.csv', use_vae=True)
	load_time = time.time() - start_time
	logger.info(f" Генератор загружен за {load_time:.2f} сек")
except Exception as e:
	logger.error(f" Ошибка загрузки генератора: {e}")
	raise


# PYDANTIC-МОДЕЛИ
class HeroRequest(BaseModel):
	"""Запрос на генерацию героя"""
	race: str = Field(..., description="Раса персонажа", example="Дварф (Холмовой)")
	class_name: str = Field(..., description="Класс персонажа", example="Друид")


class HeroResponse(BaseModel):
	"""Ответ с готовым героем"""
	Сил: int
	Лов: int
	Тел: int
	Инт: int
	Муд: int
	Хар: int
	КД: int
	МаксХП: int
	Раса: str
	Класс: str


class ErrorResponse(BaseModel):
	"""Ответ при ошибке"""
	status: str = "error"
	message: str



# ЭНДПОИНТЫ
@app.get("/")
async def root():
	"""Корневой эндпоинт — проверка работоспособности"""
	return {
		"status": "ok",
		"message": "D&D AI Character Generator API работает",
		"docs": "/docs",
		"generate": "/generate"
	}


@app.get("/health")
async def health():
	"""Эндпоинт для проверки здоровья сервера"""
	return {
		"status": "healthy",
		"model_loaded": generator is not None,
		"heroes_in_dataset": generator.data['num_heroes']
	}


@app.get("/available")
async def available_options():
	"""
	Возвращает доступные расы и классы из датасета.
	"""
	try:
		races = generator.data['df_original']['Раса'].dropna().unique().tolist()
		classes = generator.data['df_original']['Класс'].dropna().unique().tolist()

		return {
			"status": "ok",
			"races": races,
			"classes": classes
		}
	except Exception as e:
		logger.error(f"Ошибка получения опций: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=HeroResponse)
async def generate_hero(request: HeroRequest):
	"""
	Генерирует нового героя по расе и классу.

	- **race**: раса персонажа (например, "Дварф (Холмовой)")
	- **class_name**: класс персонажа (например, "Друид")
	"""
	logger.info(f" Запрос: Раса='{request.race}', Класс='{request.class_name}'")
	start = time.time()

	try:
		# Проверяем, что раса и класс существуют в датасете
		races = generator.data['df_original']['Раса'].unique().tolist()
		classes = generator.data['df_original']['Класс'].unique().tolist()

		if request.race not in races:
			raise HTTPException(
				status_code=400,
				detail=f"Раса '{request.race}' не найдена. Доступные: {races}"
			)
		if request.class_name not in classes:
			raise HTTPException(
				status_code=400,
				detail=f"Класс '{request.class_name}' не найден. Доступные: {classes}"
			)

		# Генерируем героя
		hero = generator.generate_hero(request.race, request.class_name)

		# Добавляем расу и класс в ответ
		hero['Раса'] = request.race
		hero['Класс'] = request.class_name

		# Приводим к int (на случай float)
		for key in ['Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар', 'КД', 'МаксХП']:
			if key in hero:
				hero[key] = int(round(hero[key]))

		elapsed = time.time() - start
		logger.info(f" Герой сгенерирован за {elapsed * 1000:.0f} мс")

		return hero

	except HTTPException:
		raise
	except Exception as e:
		logger.error(f" Ошибка генерации: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch")
async def generate_batch(requests: list[HeroRequest]):
	"""
	Генерирует несколько героев за один запрос.
	"""
	logger.info(f" Пакетный запрос: {len(requests)} героев")
	results = []

	for req in requests:
		try:
			hero = generator.generate_hero(req.race, req.class_name)
			hero['Раса'] = req.race
			hero['Класс'] = req.class_name
			for key in ['Сил', 'Лов', 'Тел', 'Инт', 'Муд', 'Хар', 'КД', 'МаксХП']:
				if key in hero:
					hero[key] = int(round(hero[key]))
			results.append(hero)
		except Exception as e:
			results.append({"error": str(e), "race": req.race, "class_name": req.class_name})

	return {"status": "ok", "heroes": results}


if os.path.exists("static"):
	app.mount("/static", StaticFiles(directory="static", html=True), name="static")
	logger.info(" Статика подключена: /static")