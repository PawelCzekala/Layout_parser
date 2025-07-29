init:
	python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

build:
	docker build -t layout-api .

run:
	docker run -p 8000:8000 layout-api