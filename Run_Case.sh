#export PYTHONPATH=$PWD
start_conda
source activate 2D_Turbulence
SECONDS=0
python 2D_Turbulence_ML.py
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
conda deactivate
