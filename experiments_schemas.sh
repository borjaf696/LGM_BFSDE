clear
# T 1
hatch run python scripts/launcher.py --normalize True --T 1 --phi zerobond --nsteps 48 --nsims 5000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True
# T 2
hatch run python scripts/launcher.py --normalize True --T 2 --phi zerobond --nsteps 48 --nsims 5000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True
# T 3
hatch run python scripts/launcher.py --normalize True --T 3 --phi zerobond --nsteps 48 --nsims 5000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True
# T 4
hatch run python scripts/launcher.py --normalize True --T 4 --phi zerobond --nsteps 48 --nsims 5000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True
# T 5
hatch run python scripts/launcher.py --normalize True --T 5 --phi zerobond --nsteps 48 --nsims 5000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True
# T 6
#  hatch run python scripts/launcher.py --normalize True --T 5 --phi zerobond --nsteps 48 --nsims 1000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True --simulate-in-epoch True
# T 7
# hatch run python scripts/launcher.py --normalize True --T 6 --phi zerobond --nsteps 48 --nsims 1000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True --simulate-in-epoch True
# T 8
# hatch run python scripts/launcher.py --normalize True --T 7 --phi zerobond --nsteps 48 --nsims 1000 --nepochs 100 --sigma 0.01 --schema 1 --wandb True --save True --simulate-in-epoch True
# T 8
# atch run python scripts/launcher.py --normalize True --T 8 --phi zerobond --nsteps 48 --nsims 1000 --nepochs 1500 --sigma 0.01 --schema 1 --wandb True --save True --simulate-in-epoch True
# hatch run python scripts/launcher.py --normalize True --T 4 --phi zerobond --nsteps 48 --nepochs 500 --sigma 0.01 --schema 1 --wandb True --save True
# hatch run python scripts/launcher.py --normalize True --T 5 --phi zerobond --nsteps 48 --nepochs 500 --sigma 0.01 --schema 1 --wandb True --save True
# hatch run python scripts/launcher.py --normalize True --T 6 --phi zerobond --nsteps 48 --nepochs 500 --sigma 0.01 --schema 1 --wandb True --save True
# hatch run python scripts/launcher.py --normalize True --T 7 --phi zerobond --nsteps 48 --nepochs 500 --sigma 0.01 --schema 1 --wandb True --save True
# hatch run python scripts/launcher.py --normalize True --T 8 --phi zerobond --nsteps 48 --nepochs 500 --sigma 0.01 --schema 1 --wandb True --save True
# hatch run python scripts/launcher.py --T 1 --phi zerobond --nsteps 48 --nepochs 400 --sigma 0.01 --schema 2 --wandb True
# hatch run python scripts/launcher.py --T 1 --phi zerobond --nsteps 48 --nepochs 400 --sigma 0.01 --schema 3 --wandb True