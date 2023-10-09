conda config --env --add channels conda-forge

cd /home/s9gtgros/Alia/
bash convert.sh
# python train.py > output.txt 2>&1
pip list > pip_list.txt 2>&1

conda env export --no-builds > environment_export.yml.txt 

# python train_baseline.py 2>&1 | tee output_baseline.txt
python train.py 2>&1 | tee output.txt



