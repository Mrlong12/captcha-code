from conda/miniconda3-centos7
workdir /code
copy . /code
run pip install -i  https://pypi.doubanio.com/simple -r requirements.txt
cmd ["python","run.py"]